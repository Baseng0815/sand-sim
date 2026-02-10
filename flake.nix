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
          linuxPackages.nvidiaPackages.beta
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
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
          echo "CUDA dev shell ready"
        '';
      };
    };
}
