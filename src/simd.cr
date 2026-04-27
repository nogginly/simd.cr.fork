require "log"

# Provides a standardized interface for performing SIMD operations on different CPUs
# with a scalar fallback where there is no optimal path available.
module SIMD
  {% begin %}
    VERSION = {{ `shards version "#{__DIR__}"`.chomp.stringify.downcase }}
  {% end %}

  Log = ::Log.for(self)

  @[Flags]
  enum SupportedSIMD
    SSE2
    SSE41
    AVX2
    AVX512
    NEON
    SVE
    SVE2
  end

  @[Flags]
  enum AVX512Subsets
    # Leaf 7, subleaf 0, EBX bits:
    F  # AVX-512 Foundation (base)
    DQ # Doubleword/Quadword
    CD # Conflict Detection
    BW # Byte/Word
    VL # Vector Length extensions

    # Leaf 7, subleaf 0, ECX bits:
    VBMI
    VBMI2
    VNNI
    BITALG
    VPOPCNTDQ

    # Leaf 7, subleaf 0, EDX bits (selected):
    VP2INTERSECT
    FP16
  end

  SCALAR_FALLBACK = SIMD::Scalar.new

  def self.scalar
    SCALAR_FALLBACK
  end

  {% begin %}
    class_getter instance : SIMD::Base do
      supports = supported_instruction_sets
      {% if flag?(:x86_64) %}
        case supports
        in .avx512?
          SIMD::AVX512.new
        in .avx2?
          SIMD::AVX2.new
        in .sse41?
          SIMD::SSE41.new
        in .sse2?
          SIMD::SSE2.new
        in .sve2?, .sve?, .neon?, .none?, SIMD::SupportedSIMD
          Log.trace { "falling back to scalar operations" }
          SCALAR_FALLBACK
        end
      {% elsif flag?(:aarch64) %}
        case supports
        {% unless flag?(:darwin) %}
        in .sve2?
          SIMD::SVE2.new
        in .sve?
          SIMD::SVE.new
        {% end %}
        in .neon?
          SIMD::NEON.new
        in .avx512?, .avx2?, .sse41?, .sse2?, .none?, SIMD::SupportedSIMD
          Log.trace { "falling back to scalar operations" }
          SCALAR_FALLBACK
        end
      {% else %}
        Log.trace { "falling back to scalar operations on unsupported architecture" }
        SCALAR_FALLBACK
      {% end %}
    end
  {% end %}
end

require "./simd/base"
require "./simd/detect"
require "./simd/scalar"

{% if flag?(:x86_64) %}
  require "./simd/x86_64/*"
{% elsif flag?(:aarch64) %}
  # NEON is present on all AArch64 platforms.
  require "./simd/aarch64/neon"
  # SVE/SVE2 use instructions that LLVM's integrated assembler rejects at
  # compile time on any target that doesn't advertise SVE support (e.g. Apple
  # Silicon, Raspberry Pi).  Only Linux exposes hwcap-based SVE detection and
  # only Linux AArch64 hardware ships SVE in practice, so we gate these files
  # entirely at compile time.  Attempting to compile them on Darwin or BSD
  # causes the Crystal compiler itself to crash with SIGILL.
  {% unless flag?(:darwin) %}
    require "./simd/aarch64/sve"
    require "./simd/aarch64/sve2"
  {% end %}
{% end %}
