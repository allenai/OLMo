/* Copyright 2021 Google LLC
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/* This code is almost entirely based on TODO from TODO. The original
 * program was licensed under the MIT license. We have modified it for
 * for two reasons:
 *
 * 1. The original implementation used u32 indices to point into the
 *    suffix array. This is smaller and fairly cache efficient, but here
 *    in the Real World we have to work with Big Data and our datasets
 *    are bigger than 2^32 bytes. So we have to work with u64 instead.
 *
 * 2. The original implementation had a utf8 interface. This is very
 *    convenient if you're working with strings, but we are working with
 *    byte arrays almost exclusively, and so just cut out the strings.
 *
 * When the comments below contradict these two statements, that's why.
 */


use std::borrow::Cow;
use std::fmt;
use std::iter;
use std::slice;
use std::u64;

use self::SuffixType::{Ascending, Descending, Valley};

/// A suffix table is a sequence of lexicographically sorted suffixes.
///
/// The lifetimes `'s` and `'t` (respectively) refer to the text and suffix
/// indices when borrowed.
///
/// This is distinct from a suffix array in that it *only* contains
/// suffix indices. It has no "enhanced" information like the inverse suffix
/// table or least-common-prefix lengths (LCP array). This representation
/// limits what you can do (and how fast), but it uses very little memory
/// (4 bytes per character in the text).
///
/// # Construction
///
/// Suffix array construction is done in `O(n)` time and in `O(kn)` space,
/// where `k` is the number of unique characters in the text. (More details
/// below.) The specific algorithm implemented is from
/// [(Nong et al., 2009)](https://local.ugene.unipro.ru/tracker/secure/attachment/12144/Linear%20Suffix%20Array%20Construction%20by%20Almost%20Pure%20Induced-Sorting.pdf),
/// but I actually used the description found in
/// [(Shrestha et al., 2014)](http://bib.oxfordjournals.org/content/15/2/138.full.pdf),
/// because it is much more accessible to someone who is not used to reading
/// algorithms papers.
///
/// The main thrust of the algorithm is that of "reduce and conquer." Namely,
/// it reduces the problem of finding lexicographically sorted suffixes to a
/// smaller subproblem, and solves it recursively. The subproblem is to find
/// the suffix array of a smaller string, where that string is composed by
/// naming contiguous regions of the original text. If there are any duplicate
/// names, then the algorithm proceeds recursively. If there are no duplicate
/// names (base case), then the suffix array of the subproblem is already
/// computed. In essence, this "inductively sorts" suffixes of the original
/// text with several linear scans over the text. Because of the number of
/// linear scans, the performance of construction is heavily tied to cache
/// performance (and this is why `u64` is used to represent the suffix index
/// instead of a `u64`).
///
/// The space usage is roughly `6` bytes per character. (The optimal bound is
/// `5` bytes per character, although that may be for a small constant
/// alphabet.) `4` bytes comes from the suffix array itself. The extra `2`
/// bytes comes from storing the suffix type of each character (`1` byte) and
/// information about bin boundaries, where the number of bins is equal to
/// the number of unique characters in the text. This doesn't formally imply
/// another byte of overhead, but in practice, the alphabet can get quite large
/// when solving the subproblems mentioned above (even if the alphabet of the
/// original text is very small).

#[derive(Clone, Eq, PartialEq)]
pub struct SuffixTable<'s, 't> {
    text: Cow<'s, [u32]>,
    table: Cow<'t, [u64]>,
}

impl<'s, 't> SuffixTable<'s, 't> {
    /// Creates a new suffix table for `text` in `O(n)` time and `O(kn)`
    /// space, where `k` is the size of the alphabet in the text.
    ///
    /// The table stores either `S` or a `&S` and a lexicographically sorted
    /// list of suffixes. Each suffix is represented by a 32 bit integer and
    /// is a **byte index** into `text`.
    ///
    /// # Panics
    ///
    /// Panics if the `text` contains more than `2^32 - 1` bytes. This
    /// restriction is mostly artificial; there's no fundamental reason why
    /// suffix array construction algorithm can't use a `u64`. Nevertheless,
    /// `u64` was chosen for performance reasons. The performance of the
    /// construction algorithm is highly dependent on cache performance, which
    /// is degraded with a bigger number type. `u64` strikes a nice balance; it
    /// gets good performance while allowing most reasonably sized documents
    /// (~4GB).
    pub fn new<S>(text: S) -> SuffixTable<'s, 't>
    where
        S: Into<Cow<'s, [u32]>>,
    {
        let text = text.into();
        let table = Cow::Owned(sais_table(&text));
        SuffixTable { text: text, table: table }
    }

    /// The same as `new`, except it runs in `O(n^2 * logn)` time.
    ///
    /// This is a simple naive implementation that sorts the suffixes. This
    /// tends to have lower overhead, so it can be useful when creating lots
    /// of suffix tables for small strings.
    #[doc(hidden)]
    #[allow(dead_code)]
    pub fn new_naive<S>(text: S) -> SuffixTable<'s, 't>
    where
        S: Into<Cow<'s, [u32]>>,
    {
        let text = text.into();
        let table = Cow::Owned(naive_table(&text));
        SuffixTable { text: text, table: table }
    }

    /// Creates a new suffix table from an existing list of lexicographically
    /// sorted suffix indices.
    ///
    /// Note that the invariant that `table` must be a suffix table of `text`
    /// is not checked! If it isn't, this will cause other operations on a
    /// suffix table to fail in weird ways.
    ///
    /// This fails if the number of characters in `text` does not equal the
    /// number of suffixes in `table`.
    #[allow(dead_code)]
    pub fn from_parts<S, T>(text: S, table: T) -> SuffixTable<'s, 't>
    where
        S: Into<Cow<'s, [u32]>>,
        T: Into<Cow<'t, [u64]>>,
    {
        let (text, table) = (text.into(), table.into());
        assert_eq!(text.len(), table.len());
        SuffixTable { text: text, table: table }
    }

    /// Extract the parts of a suffix table.
    ///
    /// This is useful to avoid copying when the suffix table is part of an
    /// intermediate computation.
    pub fn into_parts(self) -> (Cow<'s, [u32]>, Cow<'t, [u64]>) {
        (self.text, self.table)
    }

    /// Computes the LCP array.
    #[allow(dead_code)]
    pub fn lcp_lens(&self) -> Vec<u64> {
        let mut inverse = vec![0u64; self.text.len()];
        for (rank, &sufstart) in self.table().iter().enumerate() {
            inverse[sufstart as usize] = rank as u64;
        }
        lcp_lens_quadratic(self.text(), self.table())
        // Broken on Unicode text for now. ---AG
        // lcp_lens_linear(self.text(), self.table(), &inverse)
    }

    /// Return the suffix table.
    #[inline]
    pub fn table(&self) -> &[u64] {
        &self.table
    }

    /// Return the text.
    #[inline]
    pub fn text(&self) -> &[u32] {
        &self.text
    }

    /// Returns the number of suffixes in the table.
    ///
    /// Alternatively, this is the number of *bytes* in the text.
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns `true` iff `self.len() == 0`.
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the suffix at index `i`.
    #[inline]
    #[allow(dead_code)]
    pub fn suffix(&self, i: usize) -> &[u32] {
        &self.text[self.table[i] as usize..]
    }

    /// Returns the suffix bytes starting at index `i`.
    #[inline]
    #[allow(dead_code)]
    pub fn suffix_bytes(&self, i: usize) -> &[u32] {
        &self.text[self.table[i] as usize..]
    }

    /// Returns true if and only if `query` is in text.
    ///
    /// This runs in `O(mlogn)` time, where `m == query.len()` and
    /// `n == self.len()`. (As far as this author knows, this is the best known
    /// bound for a plain suffix table.)
    ///
    /// You should prefer this over `positions` when you only need to test
    /// existence (because it is faster).
    ///
    /// # Example
    ///
    /// Build a suffix array of some text and test existence of a substring:
    ///
    /// ```rust
    /// use suffix::SuffixTable;
    ///
    /// let sa = SuffixTable::new("The quick brown fox.");
    /// assert!(sa.contains("quick"));
    /// ```
    #[allow(dead_code)]
    pub fn contains(&self, query: &[u32]) -> bool {
        query.len() > 0
            && self
                .table
                .binary_search_by(|&sufi| {
                    self.text[sufi as usize..]
                        .iter()
                        .take(query.len())
                        .cmp(query.iter())
                })
                .is_ok()
    }

    /// Returns an unordered list of positions where `query` starts in `text`.
    ///
    /// This runs in `O(mlogn)` time, where `m == query.len()` and
    /// `n == self.len()`. (As far as this author knows, this is the best known
    /// bound for a plain suffix table.)
    ///
    /// Positions are byte indices into `text`.
    ///
    /// If you just need to test existence, then use `contains` since it is
    /// faster.
    ///
    /// # Example
    ///
    /// Build a suffix array of some text and find all occurrences of a
    /// substring:
    ///
    /// ```rust
    /// use suffix::SuffixTable;
    ///
    /// let sa = SuffixTable::new("The quick brown fox was very quick.");
    /// assert_eq!(sa.positions("quick"), &[4, 29]);
    /// ```
    #[allow(dead_code)]
    pub fn positions(&self, query: &[u32]) -> &[u64] {
        // We can quickly decide whether the query won't match at all if
        // it's outside the range of suffixes.
        if self.text.len() == 0
            || query.len() == 0
            || (query < self.suffix_bytes(0)
                && !self.suffix_bytes(0).starts_with(query))
            || query > self.suffix_bytes(self.len() - 1)
        {
            return &[];
        }

        // The below is pretty close to the algorithm on Wikipedia:
        //
        //     http://en.wikipedia.org/wiki/Suffix_array#Applications
        //
        // The key difference is that after we find the start index, we look
        // for the end by finding the first occurrence that doesn't start
        // with `query`. That becomes our upper bound.
        let start = binary_search(&self.table, |&sufi| {
            query <= &self.text[sufi as usize..]
        });
        let end = start
            + binary_search(&self.table[start..], |&sufi| {
                !self.text[sufi as usize..].starts_with(query)
            });

        // Whoops. If start is somehow greater than end, then we've got
        // nothing.
        if start > end {
            &[]
        } else {
            &self.table[start..end]
        }
    }
}

impl<'s, 't> fmt::Debug for SuffixTable<'s, 't> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "\n-----------------------------------------")?;
        writeln!(f, "SUFFIX TABLE")?;
        for (rank, &sufstart) in self.table.iter().enumerate() {
            writeln!(
                f,
                "suffix[{}] {}",
                rank,
                sufstart,
            )?;
        }
        writeln!(f, "-----------------------------------------")
    }
}

// #[allow(dead_code)]
// fn lcp_lens_linear(text: &str, table: &[u64], inv: &[u64]) -> Vec<u64> {
// // This algorithm is bunk because it doesn't work on Unicode. See comment
// // in the code below.
//
// // This is a linear time construction algorithm taken from the first
// // two slides of:
// // http://www.cs.helsinki.fi/u/tpkarkka/opetus/11s/spa/lecture10.pdf
// //
// // It does require the use of the inverse suffix array, which makes this
// // O(n) in space. The inverse suffix array gives us a special ordering
// // with which to compute the LCPs.
// let mut lcps = vec![0u64; table.len()];
// let mut len = 0u64;
// for (sufi2, &rank) in inv.iter().enumerate() {
// if rank == 0 {
// continue
// }
// let sufi1 = table[(rank - 1) as usize];
// len += lcp_len(&text[(sufi1 + len) as usize..],
// &text[(sufi2 as u64 + len) as usize..]);
// lcps[rank as usize] = len;
// if len > 0 {
// // This is an illegal move because `len` is derived from `text`,
// // which is a Unicode string. Subtracting `1` here assumes every
// // character is a single byte in UTF-8, which is obviously wrong.
// // TODO: Figure out how to get LCP lengths in linear time on
// // UTF-8 encoded strings.
// len -= 1;
// }
// }
// lcps
// }

#[allow(dead_code)]
fn lcp_lens_quadratic(text: &[u32], table: &[u64]) -> Vec<u64> {
    // This is quadratic because there are N comparisons for each LCP.
    // But it is done in constant space.

    // The first LCP is always 0 because of the definition:
    //   LCP_LENS[i] = lcp_len(suf[i-1], suf[i])
    let mut lcps = vec![0u64; table.len()];
    for (i, win) in table.windows(2).enumerate() {
        lcps[i + 1] =
            lcp_len(&text[win[0] as usize..], &text[win[1] as usize..]);
    }
    lcps
}

#[allow(dead_code)]
fn lcp_len(a: &[u32], b: &[u32]) -> u64 {
    a.iter().zip(b.iter()).take_while(|(ca, cb)| ca == cb).count() as u64
}

#[allow(dead_code)]
fn naive_table(text: &[u32]) -> Vec<u64> {
    assert!(text.len() <= u64::MAX as usize);
    let mut table = vec![0u64; text.len()];
    for i in 0..table.len() {
        table[i] = i as u64;
    }
    table.sort_by(|&a, &b| text[a as usize..].cmp(&text[b as usize..]));
    table
}

fn sais_table<'s>(text: &'s [u32]) -> Vec<u64> {
    assert!(text.len() <= u64::MAX as usize);
    let mut sa = vec![0u64; text.len()];
    let mut stypes = SuffixTypes::new(text.len() as u64);
    let mut bins = Bins::new();
    sais(&mut *sa, &mut stypes, &mut bins, &Utf8(text));
    sa
}

fn sais<T>(sa: &mut [u64], stypes: &mut SuffixTypes, bins: &mut Bins, text: &T)
where
    T: Text,
    <<T as Text>::IdxChars as Iterator>::Item: IdxChar,
{
    // Instead of working out edge cases in the code below, just allow them
    // to assume >=2 characters.
    match text.len() {
        0 => return,
        1 => {
            sa[0] = 0;
            return;
        }
        _ => {}
    }

    for v in sa.iter_mut() {
        *v = 0;
    }
    stypes.compute(text);
    bins.find_sizes(text.char_indices().map(|c| c.idx_char().1));
    bins.find_tail_pointers();

    // Insert the valley suffixes.
    for (i, c) in text.char_indices().map(|v| v.idx_char()) {
        if stypes.is_valley(i as u64) {
            bins.tail_insert(sa, i as u64, c);
        }
    }

    // Now find the start of each bin.
    bins.find_head_pointers();

    // Insert the descending suffixes.
    let (lasti, lastc) = text.prev(text.len());
    if stypes.is_desc(lasti) {
        bins.head_insert(sa, lasti, lastc);
    }
    for i in 0..sa.len() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_desc(lasti) {
                bins.head_insert(sa, lasti, lastc);
            }
        }
    }

    // ... and the find the end of each bin.
    bins.find_tail_pointers();

    // Insert the ascending suffixes.
    for i in (0..sa.len()).rev() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_asc(lasti) {
                bins.tail_insert(sa, lasti, lastc);
            }
        }
    }

    // Find and move all wstrings to the beginning of `sa`.
    let mut num_wstrs = 0u64;
    for i in 0..sa.len() {
        let sufi = sa[i];
        if stypes.is_valley(sufi) {
            sa[num_wstrs as usize] = sufi;
            num_wstrs += 1;
        }
    }
    // This check is necessary because we don't have a sentinel, which would
    // normally guarantee at least one wstring.
    if num_wstrs == 0 {
        num_wstrs = 1;
    }

    let mut prev_sufi = 0u64; // the first suffix can never be a valley
    let mut name = 0u64;
    // We set our "name buffer" to be max u64 values. Since there are at
    // most n/2 wstrings, a name can never be greater than n/2.
    for i in num_wstrs..(sa.len() as u64) {
        sa[i as usize] = u64::MAX;
    }
    for i in 0..num_wstrs {
        let cur_sufi = sa[i as usize];
        if prev_sufi == 0 || !text.wstring_equal(stypes, cur_sufi, prev_sufi) {
            name += 1;
            prev_sufi = cur_sufi;
        }
        // This divide-by-2 trick only works because it's impossible to have
        // two wstrings start at adjacent locations (they must at least be
        // separated by a single descending character).
        sa[(num_wstrs + (cur_sufi / 2)) as usize] = name - 1;
    }

    // We've inserted the lexical names into the latter half of the suffix
    // array, but it's sparse. so let's smush them all up to the end.
    let mut j = sa.len() as u64 - 1;
    for i in (num_wstrs..(sa.len() as u64)).rev() {
        if sa[i as usize] != u64::MAX {
            sa[j as usize] = sa[i as usize];
            j -= 1;
        }
    }

    // If we have fewer names than wstrings, then there are at least 2
    // equivalent wstrings, which means we need to recurse and sort them.
    if name < num_wstrs {
        let split_at = sa.len() - (num_wstrs as usize);
        let (r_sa, r_text) = sa.split_at_mut(split_at);
        sais(&mut r_sa[..num_wstrs as usize], stypes, bins, &LexNames(r_text));
        stypes.compute(text);
    } else {
        for i in 0..num_wstrs {
            let reducedi = sa[((sa.len() as u64) - num_wstrs + i) as usize];
            sa[reducedi as usize] = i;
        }
    }

    // Re-calibrate the bins by finding their sizes and the end of each bin.
    bins.find_sizes(text.char_indices().map(|c| c.idx_char().1));
    bins.find_tail_pointers();

    // Replace the lexical names with their corresponding suffix index in the
    // original text.
    let mut j = sa.len() - (num_wstrs as usize);
    for (i, _) in text.char_indices().map(|v| v.idx_char()) {
        if stypes.is_valley(i as u64) {
            sa[j] = i as u64;
            j += 1;
        }
    }
    // And now map the suffix indices from the reduced text to suffix
    // indices in the original text. Remember, `sa[i]` yields a lexical name.
    // So all we have to do is get the suffix index of the original text for
    // that lexical name (which was made possible in the loop above).
    //
    // In other words, this sets the suffix indices of only the wstrings.
    for i in 0..num_wstrs {
        let sufi = sa[i as usize];
        sa[i as usize] = sa[(sa.len() as u64 - num_wstrs + sufi) as usize];
    }
    // Now zero out everything after the wstrs.
    for i in num_wstrs..(sa.len() as u64) {
        sa[i as usize] = 0;
    }

    // Insert the valley suffixes and zero out everything else..
    for i in (0..num_wstrs).rev() {
        let sufi = sa[i as usize];
        sa[i as usize] = 0;
        bins.tail_insert(sa, sufi, text.char_at(sufi));
    }

    // Now find the start of each bin.
    bins.find_head_pointers();

    // Insert the descending suffixes.
    let (lasti, lastc) = text.prev(text.len());
    if stypes.is_desc(lasti) {
        bins.head_insert(sa, lasti, lastc);
    }
    for i in 0..sa.len() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_desc(lasti) {
                bins.head_insert(sa, lasti, lastc);
            }
        }
    }

    // ... and find the end of each bin again.
    bins.find_tail_pointers();

    // Insert the ascending suffixes.
    for i in (0..sa.len()).rev() {
        let sufi = sa[i];
        if sufi > 0 {
            let (lasti, lastc) = text.prev(sufi);
            if stypes.is_asc(lasti) {
                bins.tail_insert(sa, lasti, lastc);
            }
        }
    }
}

struct SuffixTypes {
    types: Vec<SuffixType>,
}

#[derive(Clone, Copy, Debug, Eq)]
enum SuffixType {
    Ascending,
    Descending,
    Valley,
}

impl SuffixTypes {
    fn new(num_bytes: u64) -> SuffixTypes {
        SuffixTypes { types: vec![SuffixType::Ascending; num_bytes as usize] }
    }

    fn compute<'a, T>(&mut self, text: &T)
    where
        T: Text,
        <<T as Text>::IdxChars as Iterator>::Item: IdxChar,
    {
        let mut chars = text.char_indices().map(|v| v.idx_char()).rev();
        let (mut lasti, mut lastc) = match chars.next() {
            None => return,
            Some(t) => t,
        };
        self.types[lasti] = Descending;
        for (i, c) in chars {
            if c < lastc {
                self.types[i] = Ascending;
            } else if c > lastc {
                self.types[i] = Descending;
            } else {
                self.types[i] = self.types[lasti].inherit();
            }
            if self.types[i].is_desc() && self.types[lasti].is_asc() {
                self.types[lasti] = Valley;
            }
            lastc = c;
            lasti = i;
        }
    }

    #[inline]
    fn ty(&self, i: u64) -> SuffixType {
        self.types[i as usize]
    }
    #[inline]
    fn is_asc(&self, i: u64) -> bool {
        self.ty(i).is_asc()
    }
    #[inline]
    fn is_desc(&self, i: u64) -> bool {
        self.ty(i).is_desc()
    }
    #[inline]
    fn is_valley(&self, i: u64) -> bool {
        self.ty(i).is_valley()
    }
    #[inline]
    fn equal(&self, i: u64, j: u64) -> bool {
        self.ty(i) == self.ty(j)
    }
}

impl SuffixType {
    #[inline]
    fn is_asc(&self) -> bool {
        match *self {
            Ascending | Valley => true,
            _ => false,
        }
    }

    #[inline]
    fn is_desc(&self) -> bool {
        if let Descending = *self {
            true
        } else {
            false
        }
    }

    #[inline]
    fn is_valley(&self) -> bool {
        if let Valley = *self {
            true
        } else {
            false
        }
    }

    fn inherit(&self) -> SuffixType {
        match *self {
            Valley => Ascending,
            _ => *self,
        }
    }
}

impl PartialEq for SuffixType {
    #[inline]
    fn eq(&self, other: &SuffixType) -> bool {
        (self.is_asc() && other.is_asc())
            || (self.is_desc() && other.is_desc())
    }
}

struct Bins {
    alphas: Vec<u64>,
    sizes: Vec<u64>,
    ptrs: Vec<u64>,
}

impl Bins {
    fn new() -> Bins {
        Bins {
            alphas: Vec::with_capacity(10_000),
            sizes: Vec::with_capacity(10_000),
            ptrs: Vec::new(), // re-allocated later, no worries
        }
    }

    fn find_sizes<I>(&mut self, chars: I)
    where
        I: Iterator<Item = u64>,
    {
        self.alphas.clear();
        for size in self.sizes.iter_mut() {
            *size = 0;
        }
        for c in chars {
            self.inc_size(c);
            if self.size(c) == 1 {
                self.alphas.push(c);
            }
        }
        self.alphas.sort();

        let ptrs_len = self.alphas[self.alphas.len() - 1] + 1;
        self.ptrs = vec![0u64; ptrs_len as usize];
    }

    fn find_head_pointers(&mut self) {
        let mut sum = 0u64;
        for &c in self.alphas.iter() {
            self.ptrs[c as usize] = sum;
            sum += self.size(c);
        }
    }

    fn find_tail_pointers(&mut self) {
        let mut sum = 0u64;
        for &c in self.alphas.iter() {
            sum += self.size(c);
            self.ptrs[c as usize] = sum - 1;
        }
    }

    #[inline]
    fn head_insert(&mut self, sa: &mut [u64], i: u64, c: u64) {
        let ptr = &mut self.ptrs[c as usize];
        sa[*ptr as usize] = i;
        *ptr += 1;
    }

    #[inline]
    fn tail_insert(&mut self, sa: &mut [u64], i: u64, c: u64) {
        let ptr = &mut self.ptrs[c as usize];
        sa[*ptr as usize] = i;
        if *ptr > 0 {
            *ptr -= 1;
        }
    }

    #[inline]
    fn inc_size(&mut self, c: u64) {
        if c as usize >= self.sizes.len() {
            self.sizes.resize(1 + (c as usize), 0);
        }
        self.sizes[c as usize] += 1;
    }

    #[inline]
    fn size(&self, c: u64) -> u64 {
        self.sizes[c as usize]
    }
}

/// Encapsulates iteration and indexing over text.
///
/// This enables us to expose a common interface between a `String` and
/// a `Vec<u64>`. Specifically, a `Vec<u64>` is used for lexical renaming.
trait Text {
    /// An iterator over characters.
    ///
    /// Must be reversible.
    type IdxChars: Iterator + DoubleEndedIterator;

    /// The length of the text.
    fn len(&self) -> u64;

    /// The character previous to the byte index `i`.
    fn prev(&self, i: u64) -> (u64, u64);

    /// The character at byte index `i`.
    fn char_at(&self, i: u64) -> u64;

    /// An iterator over characters tagged with their byte offsets.
    fn char_indices(&self) -> Self::IdxChars;

    /// Compare two strings at byte indices `w1` and `w2`.
    fn wstring_equal(&self, stypes: &SuffixTypes, w1: u64, w2: u64) -> bool;
}

struct Utf8<'s>(&'s [u32]);

impl<'s> Text for Utf8<'s> {
    type IdxChars = iter::Enumerate<slice::Iter<'s, u32>>;

    #[inline]
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    #[inline]
    fn prev(&self, i: u64) -> (u64, u64) {
        (i - 1, self.0[i as usize - 1] as u64)
    }

    #[inline]
    fn char_at(&self, i: u64) -> u64 {
        self.0[i as usize] as u64
    }

    fn char_indices(&self) -> iter::Enumerate<slice::Iter<'s, u32>> {
        self.0.iter().enumerate()
    }

    fn wstring_equal(&self, stypes: &SuffixTypes, w1: u64, w2: u64) -> bool {
        let w1chars = self.0[w1 as usize..].iter().enumerate();
        let w2chars = self.0[w2 as usize..].iter().enumerate();
        for ((i1, c1), (i2, c2)) in w1chars.zip(w2chars) {
            let (i1, i2) = (w1 + i1 as u64, w2 + i2 as u64);
            if c1 != c2 || !stypes.equal(i1, i2) {
                return false;
            }
            if i1 > w1 && (stypes.is_valley(i1) || stypes.is_valley(i2)) {
                return true;
            }
        }
        // At this point, we've exhausted either `w1` or `w2`, which means the
        // next character for one of them should be the sentinel. Since
        // `w1 != w2`, only one string can be exhausted. The sentinel is never
        // equal to another character, so we can conclude that the wstrings
        // are not equal.
        false
    }
}

struct LexNames<'s>(&'s [u64]);

impl<'s> Text for LexNames<'s> {
    type IdxChars = iter::Enumerate<slice::Iter<'s, u64>>;

    #[inline]
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    #[inline]
    fn prev(&self, i: u64) -> (u64, u64) {
        (i - 1, self.0[i as usize - 1])
    }

    #[inline]
    fn char_at(&self, i: u64) -> u64 {
        self.0[i as usize]
    }

    fn char_indices(&self) -> iter::Enumerate<slice::Iter<'s, u64>> {
        self.0.iter().enumerate()
    }

    fn wstring_equal(&self, stypes: &SuffixTypes, w1: u64, w2: u64) -> bool {
        let w1chars = self.0[w1 as usize..].iter().enumerate();
        let w2chars = self.0[w2 as usize..].iter().enumerate();
        for ((i1, c1), (i2, c2)) in w1chars.zip(w2chars) {
            let (i1, i2) = (w1 + i1 as u64, w2 + i2 as u64);
            if c1 != c2 || !stypes.equal(i1, i2) {
                return false;
            }
            if i1 > w1 && (stypes.is_valley(i1) || stypes.is_valley(i2)) {
                return true;
            }
        }
        // At this point, we've exhausted either `w1` or `w2`, which means the
        // next character for one of them should be the sentinel. Since
        // `w1 != w2`, only one string can be exhausted. The sentinel is never
        // equal to another character, so we can conclude that the wstrings
        // are not equal.
        false
    }
}

/// A trait for converting indexed characters into a uniform representation.
trait IdxChar {
    /// Convert `Self` to a `(usize, u64)`.
    fn idx_char(self) -> (usize, u64);
}

impl<'a> IdxChar for (usize, &'a u32) {
    #[inline]
    fn idx_char(self) -> (usize, u64) {
        (self.0, *self.1 as u64)
    }
}

impl<'a> IdxChar for (usize, &'a u64) {
    #[inline]
    fn idx_char(self) -> (usize, u64) {
        (self.0, *self.1)
    }
}

impl IdxChar for (usize, char) {
    #[inline]
    fn idx_char(self) -> (usize, u64) {
        (self.0, self.1 as u64)
    }
}

/// Binary search to find first element such that `pred(T) == true`.
///
/// Assumes that if `pred(xs[i]) == true` then `pred(xs[i+1]) == true`.
///
/// If all elements yield `pred(T) == false`, then `xs.len()` is returned.
#[allow(dead_code)]
fn binary_search<T, F>(xs: &[T], mut pred: F) -> usize
where
    F: FnMut(&T) -> bool,
{
    let (mut left, mut right) = (0, xs.len());
    while left < right {
        let mid = (left + right) / 2;
        if pred(&xs[mid]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}
