
def includes():
    out = " \
#include <torch/all.h>\n \
#include <torch/python.h>\n \
#include <omp.h>\n \
#include <cmath>\n \
#include <immintrin.h>\n \
\n \
#define mymin(a,b) ((a)<(b)?(a):(b))\n \
#define mymax(a,b) ((a)>(b)?(a):(b))\n \
"
    return out


def module(bits_list=[4, 2]):
    out = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
    for bits in bits_list:
        out += '  m.def("forward{}", &forward{}_cpu);\n'.format(bits, bits)

    for bits in bits_list:
        out += '  m.def("unpack_zeros{}", &unpack_zeros{});\n'.format(bits, bits)
    
    for bits in bits_list:
        out += '  m.def("forward_gs{}", &forward{}_gs_cpu);\n'.format(bits, bits)
    
    for bits in bits_list:
        out += '  m.def("pack{}", &pack{}_w_cpu);\n'.format(bits, bits)

    out += 'm.def("compute_reduction_cpp", &compute_reduction);\n'
    out += 'm.def("unquantize_sim", &unquantize_sim);\n'

    # if oracle:
        # out += '  m.def("forward4_oracle", &forward4_oracle_cpu);\n'


    out += 'm.def("quant_scalar_scaled", &quant_scalar_cpu);\n'

    out += '}\n'
    return out

def quant_scalar():
    out = " \
void quantize_scalar(float* A, int* BQ, float* scales, float* zeros, int n, int m, int bits){ \n \
	//find scales and zeros arrays \n \
	//quantize \n \
	int pack = 32/bits;\n \
	for (int j = 0; j < m; j++){\n \
		for (int i = 0; i < n; i+=pack){\n \
			uint32_t acc = 0;\n \
			for (int ii = i; ii < i+pack; ii++){\n \
				float ftemp = std::round((A[ii*m+j] + zeros[j])/scales[j]);\n \
				int temp = (int)ftemp;\n \
				acc = acc | (temp << (bits*(ii-i)));\n \
			}\n \
			BQ[(i/pack)*m+j] = acc;\n \
			//BQ[0] = acc;\n \
		}\n \
	}\n \
}\n \
\n \
void quant_scalar_cpu(\n \
	torch::Tensor in, torch::Tensor out, \n \
	torch::Tensor scales, torch::Tensor zeros, int bits\n \
) {\n \
\n \
	int N  = in.size(0);\n \
	int M  = in.size(1);\n \
\n \
	float* input = in.data_ptr<float>(); \n \
	float* s   = scales.data_ptr<float>();\n \
	float* z   = zeros.data_ptr<float>();\n \
	int* O   = out.data_ptr<int>();\n \
		\n \
	quantize_scalar(input, O, s, z, N, M, bits);\n \
\n \
}\n"

    return out






