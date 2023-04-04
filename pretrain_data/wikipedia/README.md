# Wikipedia

Curator: lucas@allenai.org

## Downloading Wikipedia Dumps

XML dumps are available at [dumps.wikimedia.org](https://dumps.wikimedia.org/).
As of 2023-03-25, [this page](https://en.wikipedia.org/wiki/List_of_Wikipedias#Number_of_Wikipedias_by_language_families_and_groups) lists all the languages that have a Wikipedia.

We skip Cebuano (ceb) and Swedish (sv) because they contain a [large number of machine-generated articles](https://blog.datawrapper.de/wikipedia-articles-written-by-a-bot/) of dobious quality.
The generated articles are created by [Lsjbot](<https://en.wikipedia.org/wiki/Lsjbot>).

We include simplified wikipedia (simple) because it is useful for debugging data processing scripts.

## Downloading Wikipedia Articles

We use dumps from 2020-03-20. The have the following format:

```plain-text
https://dumps.wikimedia.org/{lang_code}wiki/20230320/{lang_code}wiki-20230320-pages-articles-multistream.xml.bz2
```

where `{lang_code}` is the language code from the table above.

In order to download the dumps, first install the `pretrain_data_wikipedia` package:

```bash
pip install ./pretrain_data/wikipedia
```

Then, run the following command:

```bash
python -m pretrain_data_wikipedia.download \
    local_dst=/net/nfs2.s2-research/lucas/wikipedia \
    debug=false \
    parallel=3
```

It doesn't seem to be possible to download more than 3 in parallel. Speed seems to be limited to ~5MiB/s per connection.


## Raw Wikipedia Data

The raw Wikipedia data is available at `s3://ai2-llm/pretrain_data/wikipedia/raw/`.
License for the text data is [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
More information about the license can be found [here](https://dumps.wikimedia.org/legal.html).

## Processing wikipedia Data

Follow these steps to process the raw Wikipedia data into a format suitable for training a language model.

1. Install the `pretrain_data_wikipedia` package:

    ```bash
    pip install ./pretrain_data/wikipedia
    ```

2. Run the following command to download the raw Wikipedia data:

    ```bash
    python -m pretrain_data_wikipedia.download \
        local_dst=/net/nfs2.s2-research/lucas/wikipedia \
        debug=false \
        parallel=3
    ```

3. Use WikiExtractor to extract the text from the Wikipedia dumps:

    ```bash
    bash extract_all.sh \
        -i /net/nfs2.s2-research/lucas/wikipedia \
        -o /net/nfs2.s2-research/lucas/wikipedia-processed
    ```

4. Create version 0 of the Wikipedia data:

    ```bash
    python -m pretrain_data_wikipedia.make_v0 \
        src=/net/nfs2.s2-research/lucas/wikipedia-processed/ \
        num_workers_per_lang=12 \
        num_parallel_langs=6
    ```

## V0 Collection

Data is available at `s3://ai2-llm/pretrain_data/wikipedia/v0/`.

Load it into Athena with:

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS `llm_wikipedia_v0` (
    id STRING,
    source STRING,
    version STRING,
    text STRING,
    created STRING,
    metadata STRUCT<
        revid: STRING,
        url: STRING,
        length: BIGINT
    >
)
PARTITIONED BY (lang STRING)
ROW FORMAT serde 'org.apache.hive.hcatalog.data.JsonSerDe'
LOCATION 's3://ai2-llm/pretraining-data/sources/wikipedia/v0/documents/'
```

and then run to scan partitions:


```sql
MSCK REPAIR TABLE `llm_wikipedia_v0`
```

After loading the data into Athena, we use the following query to
obtain a count of the number of articles and tokens per language

```sql
SELECT
    lang,
    COUNT(metadata.length) AS docs_count,
    SUM(metadata.length) AS tokens_count
FROM "llm_wikipedia_v0"
GROUP BY lang
ORDER BY tokens_count DESC
```

### Statistics

Here are the numbers of documents and whitespace-separated tokens
for each language.

|**lang**         |**docs_count**|**tokens_count**|
|-----------------|--------------|----------------|
|*en*             |6,594,267     |3,040,507,044   |
|*de*             |2,761,729     |1,141,732,447   |
|*ja*             |1,358,558     |1,024,064,923   |
|*fr*             |2,434,279     |995,789,000     |
|*es*             |1,779,179     |835,797,306     |
|*ru*             |1,847,350     |730,388,102     |
|*it*             |1,617,152     |623,764,983     |
|*ceb*            |6,122,437     |598,993,318     |
|*zh*             |1,331,790     |576,634,842     |
|*pt*             |1,072,349     |375,455,775     |
|*uk*             |1,182,105     |342,950,746     |
|*nl*             |2,032,118     |342,552,753     |
|*pl*             |1,490,507     |322,702,059     |
|*ca*             |695,958       |275,823,759     |
|*sv*             |2,551,002     |254,368,784     |
|*ar*             |1,192,959     |227,133,249     |
|*vi*             |1,279,483     |212,074,253     |
|*cs*             |515,418       |179,639,179     |
|*th*             |153,359       |179,410,150     |
|*he*             |310,557       |175,815,122     |
|*hu*             |491,145       |164,439,831     |
|*fa*             |952,149       |135,805,035     |
|*no*             |599,326       |128,458,104     |
|*id*             |628,155       |127,326,993     |
|*sr*             |664,642       |126,752,653     |
|*fi*             |542,412       |119,880,566     |
|*el*             |216,403       |103,004,773     |
|*ko*             |621,477       |102,005,032     |
|*ro*             |433,284       |101,053,516     |
|*tr*             |504,327       |97,844,538      |
|*bg*             |286,229       |86,907,664      |
|*hy*             |292,333       |75,057,301      |
|*da*             |288,635       |74,233,292      |
|*gl*             |193,236       |66,174,980      |
|*eo*             |331,062       |64,890,673      |
|*ast*            |127,729       |64,836,218      |
|*eu*             |399,978       |63,499,219      |
|*sh*             |455,374       |62,844,974      |
|*arz*            |1,613,920     |60,815,855      |
|*hr*             |191,751       |58,417,520      |
|*war*            |1,266,644     |54,786,533      |
|*sl*             |179,547       |54,689,182      |
|*mk*             |132,548       |51,653,655      |
|*ms*             |362,130       |47,515,550      |
|*et*             |233,566       |47,161,814      |
|*bn*             |135,059       |46,764,668      |
|*az*             |184,412       |46,648,362      |
|*uz*             |224,633       |46,217,050      |
|*sk*             |235,051       |45,074,589      |
|*be*             |225,145       |43,894,553      |
|*hi*             |156,520       |41,984,883      |
|*cy*             |273,049       |41,350,824      |
|*lt*             |205,432       |38,563,496      |
|*my*             |106,477       |38,151,804      |
|*ur*             |185,272       |35,546,402      |
|*te*             |79,712        |35,039,382      |
|*ce*             |558,005       |34,405,232      |
|*simple*         |225,825       |33,393,601      |
|*kk*             |235,446       |33,337,263      |
|*ka*             |162,614       |32,351,732      |
|*tt*             |480,132       |30,917,410      |
|*nn*             |163,539       |30,817,747      |
|*ta*             |154,356       |30,815,347      |
|*af*             |106,511       |30,778,466      |
|*pnb*            |63,127        |30,193,612      |
|*sq*             |92,407        |27,210,085      |
|*lv*             |118,442       |25,654,663      |
|*bs*             |87,174        |23,764,507      |
|*bo*             |12,164        |21,508,208      |
|*zh_min_nan*     |427,547       |21,450,803      |
|*zh_yue*         |126,239       |20,416,833      |
|*ba*             |62,027        |19,483,080      |
|*ml*             |83,722        |18,756,348      |
|*km*             |11,453        |18,287,143      |
|*fy*             |45,678        |17,495,511      |
|*kn*             |29,335        |16,756,076      |
|*oc*             |87,989        |16,087,412      |
|*min*            |225,569       |13,819,010      |
|*pa*             |44,631        |13,606,729      |
|*ky*             |78,642        |12,482,788      |
|*ps*             |18,587        |12,368,062      |
|*la*             |134,961       |12,352,428      |
|*azb*            |241,416       |12,088,104      |
|*mr*             |82,983        |11,829,822      |
|*tl*             |43,972        |11,106,300      |
|*br*             |77,961        |10,438,009      |
|*nds*            |81,461        |10,374,452      |
|*is*             |56,504        |10,238,134      |
|*als*            |28,119        |10,049,003      |
|*ha*             |23,595        |9,366,983       |
|*tg*             |107,145       |9,216,041       |
|*mg*             |95,281        |8,778,103       |
|*sw*             |76,490        |8,646,990       |
|*lb*             |58,968        |8,296,885       |
|*jv*             |70,424        |8,290,088       |
|*new*            |71,910        |8,145,379       |
|*si*             |20,586        |7,969,402       |
|*ga*             |55,156        |7,830,802       |
|*gu*             |30,242        |7,516,294       |
|*an*             |40,547        |6,927,394       |
|*ckb*            |44,756        |6,766,468       |
|*mnw*            |2,970         |6,680,050       |
|*mn*             |22,850        |6,520,891       |
|*lmo*            |56,073        |6,356,495       |
|*io*             |37,382        |5,844,891       |
|*ig*             |14,867        |5,780,122       |
|*su*             |61,234        |5,644,053       |
|*wuu*            |40,388        |5,547,543       |
|*ne*             |31,555        |5,454,150       |
|*sco*            |36,109        |5,371,126       |
|*ku*             |58,269        |5,289,278       |
|*pms*            |67,721        |5,257,435       |
|*as*             |11,274        |4,744,821       |
|*mt*             |5,266         |4,576,457       |
|*vec*            |68,605        |4,531,660       |
|*sd*             |14,520        |4,343,635       |
|*hyw*            |10,697        |4,266,378       |
|*lld*            |125,709       |4,170,953       |
|*bar*            |26,593        |4,092,609       |
|*li*             |14,198        |4,072,138       |
|*cv*             |48,248        |3,909,726       |
|*zh_classical*   |11,769        |3,720,463       |
|*or*             |16,388        |3,673,937       |
|*shn*            |10,349        |3,430,109       |
|*ht*             |67,138        |3,359,448       |
|*ug*             |8,259         |3,283,330       |
|*yi*             |14,003        |3,170,606       |
|*sah*            |16,145        |3,060,345       |
|*mwl*            |4,228         |3,058,054       |
|*blk*            |2,572         |3,035,927       |
|*sa*             |11,876        |2,953,894       |
|*lo*             |4,871         |2,897,957       |
|*vo*             |32,821        |2,755,581       |
|*rm*             |3,802         |2,751,965       |
|*szl*            |55,931        |2,617,286       |
|*bpy*            |24,824        |2,585,573       |
|*bcl*            |13,566        |2,463,348       |
|*skr*            |5,635         |2,380,572       |
|*avk*            |24,667        |2,290,735       |
|*sat*            |7,995         |2,270,031       |
|*gom*            |4,208         |2,260,741       |
|*scn*            |23,534        |2,196,686       |
|*szy*            |4,207         |2,167,656       |
|*ia*             |25,475        |2,088,951       |
|*gd*             |15,890        |2,038,027       |
|*so*             |10,860        |1,991,905       |
|*yo*             |31,647        |1,918,624       |
|*diq*            |39,371        |1,914,947       |
|*ilo*            |14,906        |1,903,761       |
|*am*             |13,398        |1,900,731       |
|*sc*             |7,274         |1,891,133       |
|*nds_nl*         |6,625         |1,891,002       |
|*fo*             |12,330        |1,862,037       |
|*wa*             |10,836        |1,716,509       |
|*ban*            |17,802        |1,628,203       |
|*tk*             |6,780         |1,625,312       |
|*lij*            |7,869         |1,613,183       |
|*xmf*            |16,912        |1,605,272       |
|*nv*             |21,352        |1,602,175       |
|*lfn*            |4,724         |1,552,570       |
|*vls*            |7,532         |1,459,904       |
|*vep*            |6,725         |1,435,443       |
|*hsb*            |13,650        |1,352,317       |
|*dz*             |728           |1,343,342       |
|*frr*            |16,399        |1,282,690       |
|*rw*             |6,207         |1,218,634       |
|*co*             |6,474         |1,124,780       |
|*mzn*            |13,860        |1,067,757       |
|*bh*             |7,898         |1,061,703       |
|*qu*             |23,007        |1,055,142       |
|*tyv*            |3,389         |1,000,780       |
|*mai*            |14,190        |980,764         |
|*trv*            |1,806         |959,293         |
|*roa_tara*       |9,064         |917,108         |
|*tw*             |2,895         |893,249         |
|*rue*            |6,566         |876,819         |
|*lg*             |3,780         |859,563         |
|*mhr*            |10,294        |843,914         |
|*dv*             |4,307         |838,497         |
|*os*             |14,460        |833,376         |
|*nqo*            |1,561         |827,845         |
|*pam*            |8,260         |823,392         |
|*ie*             |11,562        |819,796         |
|*nap*            |12,710        |800,710         |
|*bjn*            |10,338        |798,364         |
|*ary*            |6,417         |796,657         |
|*gv*             |5,259         |772,509         |
|*cdo*            |11,053        |769,933         |
|*hak*            |9,420         |764,341         |
|*zu*             |11,166        |743,594         |
|*kab*            |5,361         |731,725         |
|*zea*            |5,554         |719,305         |
|*gn*             |5,167         |713,092         |
|*bat_smg*        |17,172        |712,306         |
|*ami*            |1,529         |710,171         |
|*kw*             |6,879         |701,053         |
|*lad*            |3,628         |693,795         |
|*crh*            |23,432        |689,695         |
|*lez*            |4,214         |687,089         |
|*dag*            |8,272         |655,787         |
|*hif*            |10,531        |655,355         |
|*wo*             |1,661         |649,508         |
|*myv*            |7,740         |646,232         |
|*fur*            |3,897         |616,972         |
|*sn*             |10,122        |604,849         |
|*stq*            |3,817         |593,468         |
|*tn*             |1,113         |588,172         |
|*glk*            |6,929         |574,974         |
|*kbp*            |1,929         |572,319         |
|*map_bms*        |11,875        |572,007         |
|*mi*             |7,815         |568,323         |
|*kaa*            |3,399         |567,676         |
|*ext*            |3,521         |556,296         |
|*pap*            |3,143         |549,608         |
|*pfl*            |2,750         |515,278         |
|*tcy*            |2,114         |502,234         |
|*gor*            |14,523        |500,508         |
|*mrj*            |10,539        |488,947         |
|*bxr*            |2,783         |487,146         |
|*gan*            |4,277         |483,334         |
|*smn*            |5,047         |475,498         |
|*ksh*            |2,755         |472,581         |
|*fiu_vro*        |6,478         |470,543         |
|*kv*             |5,582         |468,947         |
|*tay*            |2,580         |462,832         |
|*mni*            |10,794        |458,382         |
|*csb*            |5,276         |416,227         |
|*tum*            |9,286         |411,228         |
|*pcd*            |5,549         |401,245         |
|*om*             |1,425         |391,545         |
|*nrm*            |4,722         |390,319         |
|*frp*            |5,636         |358,438         |
|*olo*            |3,977         |356,266         |
|*nso*            |8,343         |350,790         |
|*eml*            |5,187         |349,667         |
|*cbk_zam*        |3,293         |346,448         |
|*dty*            |3,582         |344,069         |
|*se*             |6,633         |343,558         |
|*nah*            |7,268         |341,836         |
|*ang*            |3,890         |338,249         |
|*av*             |3,201         |332,320         |
|*udm*            |5,120         |327,825         |
|*ay*             |5,229         |327,614         |
|*dsb*            |3,245         |318,179         |
|*krc*            |1,514         |313,684         |
|*guw*            |1,282         |283,800         |
|*xh*             |1,498         |282,105         |
|*nia*            |1,649         |272,976         |
|*pcm*            |872           |262,902         |
|*shi*            |1,312         |255,484         |
|*koi*            |3,494         |254,818         |
|*gcr*            |2,382         |247,197         |
|*bug*            |15,865        |246,963         |
|*ny*             |1,111         |239,381         |
|*haw*            |2,544         |232,322         |
|*kbd*            |1,622         |220,893         |
|*tet*            |1,398         |214,536         |
|*ln*             |3,366         |212,921         |
|*awa*            |3,688         |208,335         |
|*jam*            |1,763         |197,535         |
|*ff*             |1,128         |183,362         |
|*za*             |2,238         |182,469         |
|*inh*            |1,781         |180,969         |
|*ak*             |748           |163,805         |
|*gag*            |2,581         |160,295         |
|*sm*             |1,079         |159,716         |
|*mad*            |1,070         |156,073         |
|*st*             |1,040         |149,113         |
|*pwn*            |368           |143,126         |
|*jbo*            |1,336         |137,897         |
|*pdc*            |2,091         |136,959         |
|*pag*            |2,556         |131,691         |
|*atj*            |1,950         |129,630         |
|*ts*             |668           |128,553         |
|*to*             |1,573         |127,988         |
|*mdf*            |2,923         |123,233         |
|*nov*            |1,451         |117,441         |
|*srn*            |1,205         |103,440         |
|*ee*             |1,060         |103,368         |
|*ltg*            |1,023         |101,509         |
|*kcg*            |819           |98,432          |
|*fj*             |1,275         |95,754          |
|*ks*             |2,779         |94,995          |
|*din*            |504           |93,704          |
|*xal*            |1,871         |92,709          |
|*ss*             |676           |72,212          |
|*rn*             |766           |65,387          |
|*cu*             |1,199         |63,306          |
|*bm*             |1,195         |61,719          |
|*arc*            |1,894         |61,291          |
|*tpi*            |1,384         |56,827          |
|*pnt*            |532           |56,168          |
|*ki*             |1,625         |55,038          |
|*ti*             |407           |53,321          |
|*got*            |993           |52,209          |
|*rmy*            |867           |52,087          |
|*bi*             |1,535         |48,947          |
|*ve*             |825           |45,688          |
|*ady*            |582           |43,698          |
|*kg*             |1,298         |42,521          |
|*pih*            |929           |42,131          |
|*lbe*            |1,257         |39,869          |
|*pi*             |3,004         |38,994          |
|*na*             |1,667         |34,257          |
|*chr*            |1,105         |33,759          |
|*ty*             |1,311         |32,902          |
|*kl*             |298           |27,906          |
|*ch*             |569           |25,497          |
|*iu*             |565           |16,391          |
|*ik*             |823           |15,417          |
|*sg*             |537           |12,007          |
|*chy*            |788           |9,603           |
|*cr*             |180           |3,472           |
|                 |              |                |
|**Totals**       |59,564,374    |16,088,145,777  |


## V1 Collection

...
