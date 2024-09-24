#FROM registry.suse.com/bci/bci-base:15.3.17.20.5
#FROM registry.suse.com/bci/bci-base:15.3.17.20.101
FROM registry.suse.com/bci/bci-base:15.3.17.20.145

ARG SERVER_PORT

#
# Disable BCI repros
#

RUN set -eux ; \
  sed -i 's#enabled=1#enabled=0#g' /etc/zypp/repos.d/SLE_BCI.repo 

RUN set -eux ; \
  zypper -n addrepo http://download.opensuse.org/distribution/leap/15.3/repo/oss/ myrepo1 ; \
  echo 'gpgcheck=0' >> /etc/zypp/repos.d/myrepo1.repo ; \
  zypper -n addrepo https://download.opensuse.org/repositories/devel:/languages:/perl/SLE_15_SP3 myrepo2 ; \
  echo 'gpgcheck=0' >> /etc/zypp/repos.d/myrepo2.repo
  
RUN set -eux ; \
  sed -i 's#gpgcheck=1#gpgcheck=0#g' /etc/zypp/repos.d/*.repo

#
# Install build dependencies
#
RUN set -eux; \
  zypper -n refresh ; \
  zypper --no-gpg-checks -n install -y --force-resolution \
    git cmake gcc10 gcc10-c++ gcc10-fortran zlib-devel numactl awk patch tar autoconf automake libtool libjson-c-devel graphviz ncurses-devel nano which ; \
  zypper clean

#
# Cray info
#
ENV CPE_VERSION "23.03"
ENV CPE_URL="http://localhost:$SERVER_PORT/cpe-$CPE_VERSION.tar"
ENV LIBFABRIC_VERSION "1.15.2.0" 
ENV LIBFABRIC_PATH /opt/cray/libfabric/$LIBFABRIC_VERSION
ENV MPICH_PATH "/opt/cray/pe/mpich/8.1.25/ofi/crayclang/10.0"
ENV LD_LIBRARY_PATH /opt/cray-deps:$LIBFABRIC_PATH/lib64:$MPICH_PATH/lib:/opt/cray/pe/lib64:/opt/cray/pe/lib64/cce

ENV REMOVE_CRAY_DEPS 'rm -rf /opt/cray /opt/cray-deps /usr/lib64/libcxi.so*'

ENV ROCM_RPM https://repo.radeon.com/amdgpu-install/5.5.1/sle/15.3/amdgpu-install-5.5.50501-1.noarch.rpm 
ENV ROCM_RELEASE 5.5.1

RUN set -eux ; \
  zypper --no-gpg-checks -n install $ROCM_RPM

RUN set -eux ; \
  sed -i 's#gpgcheck=1#gpgcheck=0#g' /etc/zypp/repos.d/*.repo

RUN set -eux ; \
 amdgpu-install -y --no-dkms --usecase=rocm --rocmrelease=$ROCM_RELEASE

#
# ROCm environment
#
ENV ROCM_PATH /opt/rocm-$ROCM_RELEASE
ENV PATH $ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROCM_PATH/lib

#
# Mark RCCL as non-debug - this can me overriden by RCCL debug build. 
#
ENV RCCL_DEBUG 0

#
# Fix rocm-smi lib
#
RUN set -eux ; \
  cd /opt ; \
  curl -LO http://localhost:$SERVER_PORT/rocm-5.5.1-rocm-smi-lib.patch ; \
  git clone -b rocm-$ROCM_RELEASE https://github.com/RadeonOpenCompute/rocm_smi_lib /opt/mybuild ; \
  mkdir -p /opt/mybuild/build ; \
  \
  cd /opt/mybuild ; \
  git apply < /opt/rocm-5.5.1-rocm-smi-lib.patch ; \
  rm -rf /opt/rocm-5.5.1-rocm-smi-lib.patch ; \
  \  
  cd /opt/mybuild/build ; \
  cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm-patched .. ; \
  make -j32 ; \
  make -j32 install ; \
  cd / ; rm -rf /opt/mybuild

RUN set -eux ; \
  for i in liboam librocm_smi64 ; do \
    src=$(find /opt/rocm-$ROCM_RELEASE -type f -iname $i.so*) ; \
    dst=$(find /opt/rocm-patched -type f -iname $i.so*) ; \
    rm -rf $src ; \
    ln -s $dst $src ; \
  done

RUN set -eux ; \
  cd $ROCM_PATH/bin ; \
  for i in rocm_agent_enumerator rocminfo ; do \
    rm -rf $i ; \
    curl -LO http://localhost:$SERVER_PORT/$i ; \
    chmod +x $i ; \
  done
#
# Install miniconda
#
RUN set -eux ; \
  curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh ; \
  bash ./Miniconda3-* -b -p /opt/miniconda3 -s ; \
  rm -rf ./Miniconda3-*

ENV WITH_CONDA "source /opt/miniconda3/bin/activate base"
RUN set -eux ; \
  cd / ; \
  curl -LO $CPE_URL ; \
  tar -xf *.tar ; rm -rf *.tar ; \
  \
  git clone -b cxi https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl /opt/mybuild ; \
  cd /opt/mybuild ; \
  ./autogen.sh ; \
  \
  cd /opt/mybuild ; \
  export CPATH=$LIBFABRIC_PATH/include ; \
  export LIBRARY_PATH=$LD_LIBRARY_PATH ; \
  LDFLAGS='-lcxi' CC=gcc-10 ./configure --with-libfabric=$LIBFABRIC_PATH --enable-trace --with-hip=$ROCM_PATH --with-rccl=$ROCM_PATH/rccl --disable-tests ; \
  LDFLAGS='-lcxi' CC=gcc-10 nice make -j ; \
  \
  mkdir /opt/aws-ofi-rccl ; \
  mv src/.libs/librccl-net.so* /opt/aws-ofi-rccl ; \
  rm -rf /opt/mybuild ; \
  $REMOVE_CRAY_DEPS
  
#
# Add relevant libs to execution environment
#
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/aws-ofi-rccl
ENV CXI_FORK_SAFE=1
ENV CXI_FORK_SAFE_HP=1
ENV FI_CXI_DISABLE_CQ_HUGETLB=1
RUN set -eux ; \
  cd / ; \
  curl -LO $CPE_URL ; \
  tar -xf *.tar ; rm -rf *.tar ; \
  \
  git clone https://github.com/ROCmSoftwarePlatform/rccl-tests /opt/mybuild ; \
  #git clone -b remove-stream-queries https://github.com/sfantao/rccl-tests /opt/mybuild ; \
  sed -i 's/-std=c++14/-std=c++14 --amdgpu-target=gfx90a:xnack- --amdgpu-target=gfx90a:xnack+/g' /opt/mybuild/src/Makefile ; \
  \
  cd /opt/mybuild ; \
  CC=gcc=10 \
    DEBUG=$RCCL_DEBUG \
    CXX=g++-10 \
    MPI_HOME=$MPICH_PATH \
    ROCM_PATH=$ROCM_PATH \
    MPI=1 \
    NCCL_HOME=$ROCM_PATH/rccl \
    nice make -j ; \
  mkdir /opt/rccltests ; \
  mv /opt/mybuild/build/* /opt/rccltests ; \
  rm -rf /opt/mybuild ; \
  $REMOVE_CRAY_DEPS
  
#
# Install conda environment
# 
ARG PYTHON_VERSION
RUN $WITH_CONDA; set -eux ; \
  conda create -n pytorch python=$PYTHON_VERSION ; \
  conda activate pytorch ; \
  conda install -y --only-deps pytorch ; \
  conda install -y ninja pillow mkl-include
ENV WITH_CONDA "source /opt/miniconda3/bin/activate pytorch"

#
# Install magma
# 
RUN $WITH_CONDA; set -eux ; \
  \
  export MKLROOT=$CONDA_PREFIX ; \
  \
  mkdir -p /opt/magma ; \
  git clone https://bitbucket.org/icl/magma /opt/magma-build ; \
  cd /opt/magma-build ; \
  git checkout -b mydev c62d700d880c7283b33fb1d615d62fc9c7f7ca21 ; \
  \
  cp make.inc-examples/make.inc.hip-gcc-mkl make.inc ; \
  echo 'LIBDIR += -L$(MKLROOT)/lib' >> make.inc ; \
  echo "LIB += -Wl,--enable-new-dtags -Wl,--rpath,$ROCM_PATH/lib -Wl,--rpath,\$(MKLROOT)/lib -Wl,--rpath,/opt/magma/lib" >> make.inc ; \
  echo 'DEVCCFLAGS += --gpu-max-threads-per-block=256' >> make.inc ; \
  echo 'DEVCCFLAGS += --amdgpu-target=gfx90a' >> make.inc ; \
  sed -i 's/^FOPENMP/#FOPENMP/g' make.inc ; \
  sed -i 's/VALID_GFXS = .*/VALID_GFXS = 908 90a/g' Makefile ; \
  export CC=gcc-10 ; \
  export CXX=g++-10 ; \
  export FORT=gfortran-10 ;\
  nice make -f make.gen.hipMAGMA -j32 ; \
  nice make lib/libmagma.a GPU_TARGET="gfx90a" MKLROOT=$MKLROOT -j32 ; \
  nice make lib/libmagma.so GPU_TARGET="gfx90a" MKLROOT=$MKLROOT -j32 ; \
  nice make testing/testing_dgemm GPU_TARGET="gfx90a"  MKLROOT=$MKLROOT -j32 ; \
  cp -rf include lib /opt/magma ; \
  cd / ; rm -rf /opt/magma-build

#
# Install pytorch
# 

# Repository for the wheel files
RUN set -eux ; \
  mkdir /opt/wheels

# Pytorch requires a recent cmake version
RUN $WITH_CONDA; set -eux ; \
  \
  conda install -y cmake pyyaml
  
ENV PYTORCH_ROCM_ARCH gfx90a 
ARG PYTORCH_VERSION
ARG PYTORCH_DEBUG
ARG PYTORCH_RELWITHDEBINFO
RUN $WITH_CONDA; set -eux ; \
  \
  git clone --recursive https://github.com/pytorch/pytorch /opt/pytorch-build ; \
  cd /opt/pytorch-build  ; \
  git checkout -b mydev v$PYTORCH_VERSION ; \
  git submodule sync ; \
  git submodule update --init --recursive --jobs 0 ; \
  #cd /opt ; \
  #curl -LO http://localhost:$SERVER_PORT/pytorch-v$PYTORCH_VERSION.tar ; \
  #tar -xf pytorch-v$PYTORCH_VERSION.tar ; \
  #rm -rf pytorch-v$PYTORCH_VERSION.tar ; \
  #mv pytorch-*/ pytorch-build  ; \
  \
  cd /opt/pytorch-build  ; \
  echo "${PYTORCH_VERSION}a0" > version.txt ;\
  nice python3 tools/amd_build/build_amd.py ; \
  \
  cd /opt/pytorch-build  ; \
  export CMAKE_PREFIX_PATH=$CONDA_PREFIX ; \
  DEBUG=$PYTORCH_DEBUG \
  REL_WITH_DEB_INFO=$PYTORCH_RELWITHDEBINFO \
  PYTORCH_BUILD_VERSION=$PYTORCH_VERSION \
  PYTORCH_BUILD_NUMBER=1 \
  CC=gcc-10 \
  CXX=g++-10 \
  CPATH=/usr/lib64/gcc/x86_64-suse-linux/10/include \
  LDFLAGS='-ltinfo' \
  CMAKE_MODULE_PATH=/opt/pytorch-build/cmake/Modules_CUDA_fix \
  PYTORCH_ROCM_ARCH="gfx90a" \
  RCCL_DIR=${ROCM_PATH}/rccl/lib/cmake/rccl \
  hip_DIR=${ROCM_PATH}/hip/cmake/ \
  VERBOSE=1 \
  V=1 \
  USE_ROCM=1 \
  PROCS=32 \
  numactl -C 0-31 python3 setup.py bdist_wheel ; \
  cp -rf /opt/pytorch-build/dist/* /opt/wheels ; \
  cd / ; rm -rf /opt/pytorch-build
  

RUN $WITH_CONDA; set -eux ; \
  \  
  pip install /opt/wheels/torch*.whl
  
#
# Pytorch dependees
#

# ubuntu|pytorch|apex|master|18921471b2eb8240f1787d28405b32bb93e9e671|https://github.com/ROCmSoftwarePlatform/apex
# ubuntu|pytorch|torchvision|release/0.15|fa99a5360fbcd1683311d57a76fcc0e7323a4c1e|https://github.com/pytorch/vision
# ubuntu|pytorch|torchdata|release/0.6|e1feeb2542293e42f083d24301386db6c003eeee|https://github.com/pytorch/data
# ubuntu|pytorch|torchtext|release/0.15|4571036cf66c539e50625218aeb99a288d79f3e1|https://github.com/pytorch/text
# ubuntu|pytorch|torchaudio|release/2.0|31de77dad5c89274451b3f5c4bcb630be12787c4|https://github.com/pytorch/audio
 
RUN $WITH_CONDA; set -eux ; \
  rm -rf $CONDA_PREFIX/lib/libstdc++.so* ; \
  \  
  git clone --recursive https://github.com/ROCmSoftwarePlatform/apex /opt/mybuild ; \
  cd /opt/mybuild ; \
  git checkout -b mydev 18921471b2eb8240f1787d28405b32bb93e9e671 ; \
  git submodule sync ; \
  git submodule update --init --recursive --jobs 0 ; \
  sed -i "s#/opt/rocm-5.0.0#$ROCM_PATH#g" setup.py ; \
  CC=gcc-10, CXX=g++-10 nice python setup.py bdist_wheel --cpp_ext --cuda_ext ; \
  cp -rf dist/* /opt/wheels ; \
  rm -rf /opt/mybuild 

RUN $WITH_CONDA; set -eux ; \
  \  
  git clone https://github.com/pytorch/vision /opt/mybuild ; \
  cd /opt/mybuild ; \
  git checkout -b mydev fa99a5360fbcd1683311d57a76fcc0e7323a4c1e ; \
  git submodule sync ; \
  git submodule update --init --recursive --jobs 0 ; \
  CC=gcc-10 CXX=g++-10 FORCE_CUDA=1 nice python3 setup.py bdist_wheel ; \
  cp -rf dist/* /opt/wheels ; \
  rm -rf /opt/mybuild 
  
RUN $WITH_CONDA; set -eux ; \
  \  
  git clone https://github.com/pytorch/data /opt/mybuild ; \
  cd /opt/mybuild ; \
  git checkout -b mydev e1feeb2542293e42f083d24301386db6c003eeee ; \
  git submodule sync ; \
  git submodule update --init --recursive --jobs 0 ; \
  CC=gcc-10 CXX=g++-10 FORCE_CUDA=1 nice python3 setup.py bdist_wheel ; \
  cp -rf dist/* /opt/wheels ; \
  rm -rf /opt/mybuild 
  
RUN $WITH_CONDA; set -eux ; \
  \  
  git clone https://github.com/pytorch/text /opt/mybuild ; \
  cd /opt/mybuild ; \
  git checkout -b mydev 4571036cf66c539e50625218aeb99a288d79f3e1 ; \
  git submodule sync ; \
  git submodule update --init --recursive --jobs 0 ; \
  CC=gcc-10 CXX=g++-10 FORCE_CUDA=1 nice python3 setup.py bdist_wheel ; \
  cp -rf dist/* /opt/wheels ; \
  rm -rf /opt/mybuild 
 
RUN $WITH_CONDA; set -eux ; \
  \
  git clone https://github.com/pytorch/audio /opt/mybuild ; \
  cd /opt/mybuild ; \
  git checkout -b mydev 31de77dad5c89274451b3f5c4bcb630be12787c4 ; \
  git submodule sync ; \
  git submodule update --init --recursive --jobs 0 ; \
  ln -s $ROCM_PATH/.info/version $ROCM_PATH/.info/version-dev ; \
  mkdir /opt/mytools ; \
  ln -s /usr/bin/gcc-10 /opt/mytools/gcc ; \
  ln -s /usr/bin/g++-10 /opt/mytools/g++ ; \
  PATH=$PATH:/opt/mytools CC=gcc-10 CXX=g++-10 FORCE_CUDA=1 nice python3 setup.py bdist_wheel ; \
  cp -rf dist/* /opt/wheels ; \
  rm -rf /opt/mybuild /opt/mytools
  
RUN $WITH_CONDA; set -eux ; \
  pip install /opt/wheels/apex-*.whl
RUN $WITH_CONDA; set -eux ; \
  pip install /opt/wheels/torchvision-*.whl
RUN $WITH_CONDA; set -eux ; \
  pip install /opt/wheels/torchdata-*.whl
RUN $WITH_CONDA; set -eux ; \
  pip install /opt/wheels/torchtext-*.whl
RUN $WITH_CONDA; set -eux ; \
  pip install /opt/wheels/torchaudio-*.whl
  