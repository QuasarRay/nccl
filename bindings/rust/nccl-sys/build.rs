use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=NCCL_HOME");
    println!("cargo:rerun-if-env-changed=NCCL_LIB_DIR");

    if env::var_os("CARGO_FEATURE_NO_LINK").is_some() {
        return;
    }

    if let Some(dir) = env::var_os("NCCL_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", PathBuf::from(dir).display());
    } else if let Some(home) = env::var_os("NCCL_HOME") {
        let home = PathBuf::from(home);
        for candidate in [home.join("lib"), home.join("lib64")] {
            if candidate.exists() {
                println!("cargo:rustc-link-search=native={}", candidate.display());
            }
        }
    }

    println!("cargo:rustc-link-lib=dylib=nccl");
}
