class Demongrep < Formula
  desc "Fast, local semantic code search powered by Rust"
  homepage "https://github.com/nahuelcio/demongrep"
  url "https://github.com/nahuelcio/demongrep/archive/refs/tags/v1.9.0.tar.gz"
  sha256 "6fa95fa546c67d83d61017dda8df2e29c63fdd403c379534667b9cf6f9f7a175"
  license "Apache-2.0"
  head "https://github.com/nahuelcio/demongrep.git", branch: "master"

  depends_on "rust" => :build
  depends_on "pkgconf" => :build
  depends_on "protobuf" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/demongrep --version")
  end
end
