# Office-title jurisdiction names

`territories.json` is the unmodified English territory-name data from Unicode
CLDR. `source.json` pins its source commit and SHA-256, and the accompanying
`UNICODE-LICENSE.txt` preserves its Unicode-3.0 notice.

The optional DSRs QA recipe uses two-letter territory entries and their aliases
longer than three characters, excluding the unknown-region entry. It removes
only a trailing `of [the] <territory>` or `for [the] <territory>` qualifier from
an office answer. It does not use evaluation references, infer new facts or
change the exact-match scorer. Unknown jurisdictions and integral title words
remain intact. The raw revised answer is retained alongside the canonical answer.
