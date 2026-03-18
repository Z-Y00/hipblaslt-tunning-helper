# latest hipBLASLt
ARG HIPBLASLT_REPO=https://github.com/ROCm/rocm-libraries.git
ARG HIPBLASLT_BRANCH=b3db63927d3df09ec2f93d46e733d7a0ab51b87b
ENV ROCM_PATH=/opt/rocm-7.2.1
ENV PYTORCH_ROCM_ARCH="gfx942;gfx950"
# We can force the install to /opt/rocm-7.2.0 using the -DCMAKE_INSTALL_PREFIX flag
RUN echo $(dpkg -l | awk '$2=="hipblaslt" { print $3 }') > ./OLD_HIPBLAST_VERSION
RUN git clone ${HIPBLASLT_REPO} \
    && cd rocm-libraries/projects/hipblaslt \
    && git checkout ${HIPBLASLT_BRANCH} \
    && python3 -m pip install -r tensilelite/requirements.txt \
    && wget -nv https://download.amd.com/developer/eula/aocl/aocl-4-2/aocl-linux-gcc-4.2.0_1_amd64.deb \
    && apt install ./aocl-linux-gcc-4.2.0_1_amd64.deb \
    && rm aocl-linux-gcc-4.2.0_1_amd64.deb \
    && dpkg --remove --force-depends hipblaslt hipblaslt-dev \
    && cmake --preset rocm-7.0.0 \
        -DGPU_TARGETS="${PYTORCH_ROCM_ARCH}" \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH}/lib/llvm;${ROCM_PATH}" \
        -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
        -DCMAKE_PACKAGING_INSTALL_PREFIX="${ROCM_PATH}" \
        -DROCM_PATH="${ROCM_PATH}" \
        -B build -S. \
    && cmake --build build --target package --parallel \
    && dpkg -i build/*.deb \
    && cd ../../../ \
    && rm -rf rocm-libraries \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache

# Link the latest hipblaslt with 0.x.x
# This is a workaround before pytorch officially supports hipblaslt 1.x.x
RUN sudo ln -s /opt/rocm/lib/libhipblaslt.so /opt/rocm/lib/libhipblaslt.so.0

# Fixes hipblaslt version in apt
RUN echo $(dpkg -l | awk '$2=="hipblaslt" { print $3 }') > ./NEW_HIPBLAST_VERSION \
    && sed -i -e "s/hipblaslt (= $(cat ./OLD_HIPBLAST_VERSION))/hipblaslt (= $(cat ./NEW_HIPBLAST_VERSION))/g" /var/lib/dpkg/status \
    && sed -i -e "s/hipblaslt-dev (= $(cat ./OLD_HIPBLAST_VERSION))/hipblaslt-dev (= $(cat ./NEW_HIPBLAST_VERSION))/g" /var/lib/dpkg/status \
    && rm OLD_HIPBLAST_VERSION NEW_HIPBLAST_VERSION
