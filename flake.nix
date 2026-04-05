{
            description = "Pong-Bot — motor control, computer vision, and RL pipeline";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        isPi = system == "aarch64-linux";
        isX86 = system == "x86_64-linux";

        # ---------------------------------------------------------------------------
        # Pi-only camera package overrides.
        # Applied only on aarch64-linux via python.override below.
        #
        # Packages:
        #   libcameraPy  — thin wrapper exposing pkgs.libcamera's Python bindings
        #                  as a proper Python package dependency.
        #   pykms-stub   — headless stub satisfying picamera2's pykms import
        #                  without requiring the Pi OS DRM preview package.
        #   pidng        — DNG file creation; has a C extension, use setuptools.
        #   simplejpeg   — JPEG encode/decode; pure wheel.
        #   videodev2    — V4L2 ctypes bindings; pure wheel.
        #   picamera2    — main camera library; pure wheel.
        # ---------------------------------------------------------------------------
        piCameraOverrides = pyFinal: pyPrev: {

          # Disable flaky aarch64 TSC test in python-prctl.
          "python-prctl" = pyPrev."python-prctl".overridePythonAttrs (_: {
            doCheck = false;
          });

          # Wrap pkgs.libcamera's site-packages as a proper Python package so
          # it can be a real propagatedBuildInput of picamera2.
          # Using buildPythonPackage ensures withPackages includes it in the
          # Python environment path correctly.
          libcameraPy = pyFinal.buildPythonPackage {
            pname = "libcamera";
            version = pkgs.libcamera.version;
            format = "other";

            dontUnpack = true;

            installPhase = ''
              sp=$out/lib/python${pyFinal.python.pythonVersion}/site-packages
              mkdir -p $sp
              cp -r ${pkgs.libcamera}/lib/python${pyFinal.python.pythonVersion}/site-packages/libcamera $sp/
            '';

            # Bring native libcamera .so into the closure.
            propagatedBuildInputs = [ pkgs.libcamera ];

            doCheck = false;
          };

          # Headless stub for pykms / kms — satisfies picamera2's preview import
          # without needing the Pi OS DRM package.  Raises RuntimeError if any
          # preview method is actually called (which we never do in headless mode).
          "pykms-stub" = pyFinal.buildPythonPackage {
            pname = "pykms-stub";
            version = "0.1.0";
            format = "other";

            dontUnpack = true;

            installPhase = ''
                            sp=$out/lib/python${pyFinal.python.pythonVersion}/site-packages
                            mkdir -p $sp

                            cat > $sp/pykms.py <<'EOF'
              # Headless stub — pykms/kms DRM preview backend is not available.
              # Attribute access returns silent sentinel values so picamera2's
              # drm_preview.py class body evaluates without error at import time.
              # Any attempt to actually *use* a DRM preview at runtime will fail
              # with a clear message.

              class _Sentinel:
                  """Returned for any attribute lookup on the stub modules."""
                  def __getattr__(self, name):
                      return _Sentinel()
                  def __call__(self, *a, **kw):
                      return _Sentinel()
                  def __int__(self):
                      return 0
                  def __index__(self):
                      return 0
                  def __repr__(self):
                      return "<pykms-stub sentinel>"

              import sys as _sys

              class _StubModule:
                  """Stub module replacing pykms / kms."""
                  _sentinel = _Sentinel()

                  def __getattr__(self, name):
                      return self._sentinel

                  def __call__(self, *a, **kw):
                      raise RuntimeError(
                          "pykms is not available in headless mode. "
                          "Use NullPreview or no preview."
                      )

              _stub_module = _StubModule()
              _sys.modules["kms"]   = _stub_module
              _sys.modules["pykms"] = _stub_module
              EOF
            '';

            doCheck = false;
          };

          pidng = pyFinal.buildPythonPackage rec {
            pname = "pidng";
            version = "4.0.9";
            format = "setuptools";

            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/c0/65/2670c465c8a63a23eb3a5e5547262e247e1aa2d3889a0a6781da9109d5f7/pidng-4.0.9.tar.gz";
              hash = "sha256-Vg6wCAhvinFf2eGrmYgXp9TIUAp/Fhuc5q9asnUB+Cw=";
            };

            nativeBuildInputs = [ pyFinal.setuptools ];
            propagatedBuildInputs = [ pyFinal.numpy ];
            doCheck = false;
          };

          simplejpeg = pyFinal.buildPythonPackage rec {
            pname = "simplejpeg";
            version = "1.9.0";
            format = "wheel";

            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/88/8b/d8ca384f1362371d61690d7460d3ae4cec4a5a25d9eb06cd15623de3725a/simplejpeg-1.9.0-cp313-cp313-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl";
              hash = "sha256-oMN1Ew9zuwgimj3tOS2E7i2Raz6H5+xdKsTke3FENGo=";
            };

            propagatedBuildInputs = [ pyFinal.numpy ];
            doCheck = false;
          };

          videodev2 = pyFinal.buildPythonPackage rec {
            pname = "videodev2";
            version = "0.0.4";
            format = "wheel";

            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/68/30/4982441a03860ab8f656702d8a2c13d0cf6f56d65bfb78fe288028dcb473/videodev2-0.0.4-py3-none-any.whl";
              hash = "sha256-0196s53bBtUP7Japm/yNW4tSW8fqA3iCWdOGOT8aZLo=";
            };

            doCheck = false;
          };

          picamera2 = pyFinal.buildPythonPackage rec {
            pname = "picamera2";
            version = "0.3.34";
            format = "wheel";

            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/57/eb/ef4ec16258285093e90d26195deb58d5b473cd334f713c2d4691955eaafe/picamera2-0.3.34-py3-none-any.whl";
              hash = "sha256-CGNckCtDz6Y0my3sRgAF+Sv02XZVyvPNV7Bvw7uLGcU=";
            };

            propagatedBuildInputs = [
              pyFinal.libcameraPy
              pyFinal."pykms-stub"
              pyFinal.numpy
              pyFinal.pidng
              pyFinal.piexif
              pyFinal.pillow
              pyFinal.simplejpeg
              pyFinal.videodev2
              pyFinal."python-prctl"
              pyFinal.av
              pyFinal."libarchive-c"
              pyFinal.tqdm
              pyFinal.jsonschema
            ];

            # openexr is only needed for the IMX500 AI camera device driver,
            # not for standard capture. Skip the runtime dep check.
            dontCheckRuntimeDeps = true;
            doCheck = false;
          };
        };

        # ---------------------------------------------------------------------------
        # Per-system Python interpreter.
        # On Pi: includes camera package overrides.
        # On x86: plain python3.
        # ---------------------------------------------------------------------------
        python =
          if isPi then pkgs.python3.override { packageOverrides = piCameraOverrides; } else pkgs.python3;

        # ---------------------------------------------------------------------------
        # qwen-vl-utils — not yet in nixpkgs, packaged from PyPI wheel.
        # Uses python.pkgs to stay consistent with the per-system interpreter.
        # ---------------------------------------------------------------------------
        qwenVlUtils = python.pkgs.buildPythonPackage {
          pname = "qwen-vl-utils";
          version = "0.0.14";
          format = "wheel";

          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/py3/q/qwen_vl_utils/qwen_vl_utils-0.0.14-py3-none-any.whl";
            sha256 = "5e28657bfd031e56bd447c5901b58ddfc3835285ed100f4c56580e0ade054e96";
          };

          propagatedBuildInputs = with python.pkgs; [
            av
            pillow
            requests
          ];

          dontCheckRuntimeDeps = true;
          doCheck = false;

          meta.description = "Utilities for Qwen-VL vision-language models";
        };

        # ---------------------------------------------------------------------------
        # Package list functions — return lists so they can be combined cleanly.
        #
        # makeBaseRuntimePyPkgs — core Pi runtime deps (both arches)
        # makePiCameraPyPkgs    — Pi-only camera deps (aarch64 only)
        # makeRuntimePyPkgs     — combined runtime deps
        # makeTrainingPyPkgs    — additional deps for training commands (x86 server)
        # ---------------------------------------------------------------------------
        makeBaseRuntimePyPkgs = with python.pkgs; [
          pyserial
          click
          opencv4
          numpy
          pip
          virtualenv
        ];

        makePiCameraPyPkgs = if isPi then [ python.pkgs.picamera2 ] else [ ];

        makeRuntimePyPkgs = makeBaseRuntimePyPkgs ++ makePiCameraPyPkgs;

        makeTrainingPyPkgs =
          let
            torch = if isX86 then python.pkgs.torchWithCuda else python.pkgs.torch;
            torchvision = python.pkgs.torchvision.override { inherit torch; };
            transformers = python.pkgs.transformers.override { inherit torch; };
            accelerate = python.pkgs.accelerate.override { inherit torch; };
            torchinfo = python.pkgs.torchinfo.override { inherit torch torchvision; };
          in
          [
            accelerate
            torch
            torchvision
            transformers
            qwenVlUtils

            python.pkgs.av
            python.pkgs.matplotlib
            python.pkgs.onnx
            python.pkgs.onnxscript
            python.pkgs.pillow
            python.pkgs.scikit-learn
            python.pkgs.scipy
            torchinfo
          ];

        # ---------------------------------------------------------------------------
        # Package constructor functions
        # ---------------------------------------------------------------------------

        makePongBotPkg = python.pkgs.buildPythonPackage {
          pname = "pong-bot";
          version = "0.3.0";
          src = ./.;

          pyproject = true;
          doCheck = false;

          build-system = [ python.pkgs.hatchling ];

          dependencies = makeRuntimePyPkgs;

          # Generate bash completion scripts for all pong-* commands and install
          # them into $out/share/bash-completion/completions/ so they are picked
          # up automatically when the package is installed via nix profile install.
          # No ~/.bashrc changes needed on the target system.
          postInstall = ''
            mkdir -p $out/share/bash-completion/completions
            _PONG_MOTOR_COMPLETE=bash_source      $out/bin/pong-motor      > $out/share/bash-completion/completions/pong-motor
            _PONG_COLLECT_CV_COMPLETE=bash_source $out/bin/pong-collect-cv > $out/share/bash-completion/completions/pong-collect-cv
            _PONG_SHOOT_COMPLETE=bash_source      $out/bin/pong-shoot      > $out/share/bash-completion/completions/pong-shoot
            _PONG_TUNE_COMPLETE=bash_source       $out/bin/pong-tune       > $out/share/bash-completion/completions/pong-tune
            _PONG_CV_SHOOT_COMPLETE=bash_source   $out/bin/pong-cv-shoot   > $out/share/bash-completion/completions/pong-cv-shoot
          '';

          meta = {
  description = "Pong-Bot — motor control, computer vision, and RL pipeline";
            mainProgram = "pong-motor";
          };
        };

        makePongBotTrainingPkg = python.pkgs.buildPythonPackage {
          pname = "pong-bot-training";
          version = "0.3.0";
          src = ./.;

          pyproject = true;
          doCheck = false;

          build-system = [ python.pkgs.hatchling ];

          dependencies = makeRuntimePyPkgs ++ makeTrainingPyPkgs;

          meta = {
            description = "Pong-Bot — full pipeline with training deps";
            mainProgram = "pong-motor";
          };
        };

        # ---------------------------------------------------------------------------
        # DevShell constructor
        # ---------------------------------------------------------------------------
        makeDevShell =
          {
            pyPkgs,
            name,
            banner,
            extraPkgs ? [ ],
          }:
          pkgs.mkShell {
            inherit name;

            packages = [
              (python.withPackages (_: pyPkgs ++ [ python.pkgs.ipython ]))
              pkgs.timg
            ]
            ++ extraPkgs;

            shellHook = ''
              export PYTHONPATH="$(git rev-parse --show-toplevel 2>/dev/null || pwd):$PYTHONPATH"

              # Create .venv-training for pip-only packages (e.g. roboflow).
              # Uses --system-site-packages so all Nix deps are visible.
              VENV=".venv-training"
              if [ ! -d "$VENV" ]; then
                echo "Creating $VENV and installing pip-only packages..."
                python -m venv --system-site-packages "$VENV" 2>/dev/null || true
                "$VENV/bin/pip" install --quiet "roboflow>=1.2" 2>/dev/null || true
              fi
              if [ -f "$VENV/bin/activate" ]; then
                source "$VENV/bin/activate"
              fi

              echo ""
              ${banner}
              echo ""
              echo "  (install with 'nix profile install .' for pong-* commands on PATH)"
              echo ""
            '';
          };

      in
      {
        # ---------------------------------------------------------------------------
        # Packages
        # ---------------------------------------------------------------------------
        packages = {
          default = makePongBotPkg;
          pong-bot = makePongBotPkg;
        }
        // pkgs.lib.optionalAttrs isX86 {
          # Training package pulls in torch/cuda — x86 server only.
          training = makePongBotTrainingPkg;
        };

        # ---------------------------------------------------------------------------
        # DevShells
        # ---------------------------------------------------------------------------
        devShells = {
          default = makeDevShell {
            name = "pong-pi-dev-env";
            pyPkgs = makeRuntimePyPkgs;
            banner = ''
              echo "pong-pi dev shell"
              echo ""
              echo "── Motor control ──────────────────────────────────────────────"
              echo "  python -m motor_control.cli --help"
              echo "  python -m motor_control.cli --dry-run fire"
              echo "  python -m motor_control.cli steps -a Y -s -500"
              echo "  python -m motor_control.cli limit-status --watch --count 20"
              echo "  python -m motor_control.cli test-home-y"
              echo "  python -m motor_control.cli home-y"
              echo "  python -m motor_control.cli set-home"
              echo "  python -m motor_control.cli home"
              echo "  python -m motor_control.cli info"
              echo ""
              echo "── Data collection ────────────────────────────────────────────"
              echo "  python -m vision.collect_shots --x-steps 1200 --y-steps 28500"
              echo "  python -m vision.collect_shots --x-steps 0 --y-steps 0 --dry-run"
              echo "  python -m vision.collect_shots --x-steps 0 --y-steps 0 --no-home-y"
              echo ""
              echo "── RL shot (Pi side — server must have StreamReceiver running) "
              echo "  python -m rl.shoot --x-steps 1200 --y-steps 28500 --server-host <host>"
              echo "  python -m rl.shoot --x-steps 0 --y-steps 0 --server-host <host> --dry-run"
              echo ""
              echo "── Autonomous run ─────────────────────────────────────────────"
              echo "  python -m run --dry-run"
              echo "  python -m run --loop --loop-delay 5"
            '';
          };
        }
        // pkgs.lib.optionalAttrs isX86 {
          # Training devshell pulls in torch/cuda — x86 server only.
          training = makeDevShell {
            name = "pong-training";
            pyPkgs = makeRuntimePyPkgs ++ makeTrainingPyPkgs;
            banner = ''
              echo "pong-training shell (GPU)"
              echo "  python -m vision.train_detector --data data/ --epochs 50"
              echo "  python -m vision.train_head --backbone models/cup_detector.pt --shots data/shots.jsonl"
              echo "  python -m rl.tune --policy heuristic --pi-host pong-pi --server-host my-server"
              python -c "import torch; print(f'  torch {torch.__version__}  cuda={torch.cuda.is_available()}')" 2>/dev/null || true
            '';
          };
        };

      }
    )
    // {
      # -------------------------------------------------------------------------
      # x86_64-linux-only Hydra jobs — training deps (torch/cuda) do not build
      # on aarch64-linux so these are declared outside eachSystem.
      # The per-system base hydraJobs are merged in from eachSystem via the
      # system-keyed attribute set built with builtins.listToAttrs below.
      # -------------------------------------------------------------------------
      hydraJobs =
        # Collect per-system base jobs from eachSystem into flat keyed attrs
        builtins.foldl'
          (
            acc: system:
            acc
            // {
              "pong-bot.${system}" = self.packages.${system}.default;
              "devShell.${system}" = self.devShells.${system}.default;
            }
          )
          { }
          [
            "x86_64-linux"
            "aarch64-linux"
          ]
        // {
          # x86-only training jobs
          "pong-bot-training.x86_64-linux" = self.packages.x86_64-linux.training;
          "devShell-training.x86_64-linux" = self.devShells.x86_64-linux.training;
        };
    };
}
