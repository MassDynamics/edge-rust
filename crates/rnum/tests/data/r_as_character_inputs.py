"""Writes column 1 of r_as_character.tsv: the bits of every test double, as 16 hex digits.

Run from this directory, then fill column 2 with R's as.character() in the production image:
    uv run python r_as_character_inputs.py
    docker run --rm --platform linux/amd64 -v "$PWD":/w -w /w md-flexi-r45-local:latest \
        Rscript r_as_character.R
Column 2 is left empty here; r_as_character.R rewrites every line.
"""

import struct

import numpy as np

# Hand cases: the fixed/scientific width rule, 15-digit rounding, extremes and signs.
v = [1e5, 110000.0, 1e-4, 1e-5, 1e15, 1e16, 1234567890123456.0, 123456789012345.0, 0.1 + 0.2, 0.3,
     1.0, 1.000000000000001, 99999.99999999999, 999999999999999.9, 9999999999999998.0, 0.0001,
     0.00012, 123456.7, 1e-300, 5e-324, 1.7976931348623157e308, 1e22, 1e23, 1e100, 1e-100, 1e-99,
     2.5, -2.5, -1e5, -0.001, 100000.5, 1e5 + 1, 3e5, 2e5, 1e6, 0.5, 12345678901234567890.0, 1 / 3,
     2 / 3, 100.0, 1000.0, 10000.0]  # fmt: skip

# Powers of ten from 1e-30 to 1e29, their neighbours and 15th-digit near-ties.
for k in range(-30, 30):
    p = 10.0**k
    v += [p, np.nextafter(p, 0), np.nextafter(p, np.inf), 2 * p, 1.5 * p, 9.5 * p,
          9.99999999999999 * p, 9.999999999999999 * p]  # fmt: skip

# Exact ties for sprintf's own rounding, and binary fractions.
v += [2.0**50 + 0.5, 2.0**51 + 0.5, 2.0**50 + 2.5, 1125899906842624.5, 0.125, 0.375, 2.0**-20,
      2.0**-40, 3 * 2.0**-30, 1e15 + 0.5, 4503599627370495.5, 9007199254740993.0, 2.0**60, 2.0**70,
      2.0**-1074 * 3]  # fmt: skip


# Random values shaped like what the engines format, in two seeded blocks that R has already
# formatted: the review r2 pool (65,000 values) and fix round 0's smaller block (5,000), so no
# R-checked row is dropped (review overnight r1, SE-m1).
def random_values(n_p, n_ave, n_lfc, n_scaled, n_rounded, n_int, n_bin):
    rng = np.random.default_rng(20261003)
    r = list(10 ** rng.uniform(-300, 0, n_p))  # p-values
    r += list(rng.lognormal(3, 4, n_ave))  # AveExpr-like
    r += list(rng.normal(0, 3, n_lfc))  # log2FC
    r += list(
        rng.integers(0, 10**7, n_scaled).astype(float) * 10.0 ** rng.integers(-6, 12, n_scaled)
    )
    r += [
        round(float(a), int(b))
        for a, b in zip(rng.uniform(-1e6, 1e6, n_rounded), rng.integers(0, 6, n_rounded))
    ]
    r += [float(a) for a in rng.integers(1, 10**6, n_int)]
    r += list(rng.integers(1, 2**52, n_bin) * 2.0 ** rng.integers(-1074, 970, n_bin))
    return r


v += random_values(20000, 20000, 10000, 5000, 5000, 3000, 2000)
v += random_values(1500, 1500, 750, 400, 400, 250, 200)

hexes = [struct.pack(">d", float(x)).hex() for x in v]

# Review deseq2 r3, R3-m1: values that the 1e22 table rounded differently from R's tbl[] to 1e27
# (42 adversarial near-ties and 84 p-value-like values, |x| in [1e-13, 1e-8) or [1e37, 1e42)).
# fmt: off
hexes += [
    "3da5f87281e000ff", "3d8755f4b9057ee8", "3da4833c7d743562", "3da2bb81abd4a4f8",
    "3db929b83426edf2", "3dd70a077b6c89ca", "3dbee8cc35d004fb", "3db00581ca78af3e",
    "3dd2c37b13c5d2d1", "3e0fd97fa94753f1", "3e3da0af866a26ad", "3e30ed3ae59f7a69",
    "3e3747aa641dfd62", "3e419bffba8c6c41", "3e39dd7da7912b4b", "482707d6167d817c",
    "3da5fd7fe1796492", "479e17b843576917", "3e431affe33395e5", "3db9b86c9fc4b0e4",
    "3dda7eca847f6b34", "bda5f87281e000ff", "bd8755f4b9057ee8", "bda4833c7d743562",
    "bda2bb81abd4a4f8", "bdb929b83426edf2", "bdd70a077b6c89ca", "bdbee8cc35d004fb",
    "bdb00581ca78af3e", "bdd2c37b13c5d2d1", "be0fd97fa94753f1", "be3da0af866a26ad",
    "be30ed3ae59f7a69", "be3747aa641dfd62", "be419bffba8c6c41", "be39dd7da7912b4b",
    "c82707d6167d817c", "bda5fd7fe1796492", "c79e17b843576917", "be431affe33395e5",
    "bdb9b86c9fc4b0e4", "bdda7eca847f6b34", "3e1606096a79cb1d", "3d57f19cd2e661c5",
    "3deacd9b3d07da6e", "3e2781595ddf1959", "3e1ce0f80df4c112", "3dc809ed22740079",
    "3dd9e9011cce7928", "3e367600e3138b0b", "3e2bfd53b3f4f284", "3da18804b779499b",
    "3da9a22370420df0", "3dd8f244c4362b4e", "3da00ea4c2b6c4c9", "3e41c4b56e3ed429",
    "3db77f91c511fe7f", "3da3077e8d9a8363", "3da398a15854b9e5", "3dd73ed10639c6a1",
    "3e3fe49e08d1f118", "3dd2cd442ca1a0a7", "3dabb0587598813c", "3d6452e02045e91f",
    "3dab14ca2dd4ef62", "3d9334ce03ebb64d", "3da8920aa214874a", "3e225ebd6d66fcad",
    "3e39e5147aac587b", "3e427866963fad16", "3e3997d0980ffa1d", "3e01703f25cd7f71",
    "3dbc3296a5ea347b", "3e45428b7cd0e531", "3dcd9d9181e3b7e4", "3e308dd5c5e12c29",
    "3e2b9bf8021edb9d", "3e3e618d38f8493b", "3e139fa88e70479a", "3dce6a3676472e8b",
    "3e3229acc87b94c5", "3e2a9cae086b7c54", "3e104c0449fcad47", "3dd4ad0cab6bac03",
    "3d9a119ce566c6c0", "3d48237091e9dd8e", "3e345444ee2e4ccf", "3d90e5dffe8857f6",
    "3dd8fdb04ccc61b5", "3d942ce791840353", "3d9e13dc59a75e30", "3d65717db021e1ae",
    "3dc761b80892fcfd", "3dc3e77c3d4156da", "3dd3c01740163437", "3e0df27a38bfc6b1",
    "3dc7294fb775eb9d", "3da0b6ababb4d45a", "3dd85f63d8d3f69b", "3e444773612061e4",
    "3d8dbb957370bbd4", "3e38ed56efeae615", "3dcbb56a3340cb7e", "3d6bd992cabad884",
    "3da2b8512a05dc83", "3dc1e7c83fee9d0e", "3e412dca9b27090b", "3dd242bb2b676210",
    "3dc214422d9e6f86", "3da3bf768a715b3b", "3dc02ee9da639e2b", "3d7008a3b363f621",
    "3dd79416d68c3882", "3e3a9274bdaf0f4f", "3e3afc8505d7b134", "3e0591fa3a35c429",
    "3db6f46f43cfded0", "3e3ea835ec4bdeee", "3e4179bbac63d84f", "3dd87e6783267ca3",
    "3da2ddd8e239e1f7", "3d76ac64c86045c6", "3df578df480aa455", "3dcf4cc7238a26e3",
    "3e41a4e14bb21d7c", "3e2306079d112323",
]
# fmt: on

with open("r_as_character.tsv", "w") as f:
    f.writelines(h + "\t\n" for h in dict.fromkeys(hexes))  # drop repeats, keep the first
