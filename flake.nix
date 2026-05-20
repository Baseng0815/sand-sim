{
  description = "CUDA development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:

    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell rec {
        nativeBuildInputs = with pkgs; [
            llvmPackages.llvm
            llvmPackages.clang-tools
            cmake
            ninja
            pkg-config
            cudaPackages.nsight_systems
        ];

        buildInputs = with pkgs; [
          # NOTE: do NOT add linuxPackages.nvidiaPackages.* here. The userspace
          # libcuda.so must match the running kernel module; bundling a driver
          # package causes "CUDA driver/library version mismatch". The matching
          # libcuda.so is provided by the host at /run/opengl-driver/lib (see
          # shellHook below).
          sdl3
          sdl3-ttf
          harfbuzz
          freetype
          glib
          libsysprof-capture
          pcre2
          cudaPackages.cudatoolkit
          cudaPackages.cuda_cudart
          cudaPackages.cuda_cupti
          cudaPackages.cuda_nvrtc
          cudaPackages.cuda_nvtx
          cudaPackages.cudnn
          cudaPackages.libcublas
          cudaPackages.libcufft
          cudaPackages.libcurand
          cudaPackages.libcusolver
          cudaPackages.libcusparse
          cudaPackages.libnvjitlink
          cudaPackages.nccl
        ];

          # export CC=${pkgs.gcc13}/bin/gcc
          # export CXX=${pkgs.gcc13}/bin/g++
          # export PATH=${pkgs.gcc13}/bin:$PATH
        shellHook = ''
          export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
          # /run/opengl-driver/lib first: the kernel-matched libcuda.so / NVML
          # live there and must take precedence over any CUDA stub libs.
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
          zsh
          echo "CUDA dev shell ready"
        '';
      };
    };
}
