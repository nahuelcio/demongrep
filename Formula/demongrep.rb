class Demongrep < Formula
  desc "Fast, local semantic code search powered by Rust"
  homepage "https://github.com/nahuelcio/demongrep"
  url "https://github.com/nahuelcio/demongrep/archive/refs/tags/v1.9.1.tar.gz"
  sha256 "1fdf2e8cdacdaeca41004edbdebfaf0fe0cc831e2f086770d81a199356661f00"
  license "Apache-2.0"
  head "https://github.com/nahuelcio/demongrep.git", branch: "master"

  depends_on "rust" => :build
  depends_on "pkgconf" => :build
  depends_on "protobuf" => :build
  depends_on "openssl" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/demongrep --version")
  end
end
