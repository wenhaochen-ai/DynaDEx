# Schema files

Each flat schema is an array of `role` and `def` entries. A definition may be a
string or an array of equivalent wordings. Training augmentation selects one
wording per label and uses the same selection in both conditioning channels.
Evaluation uses the first wording, with a fixed label order.

`framenet17.json` indexes role definitions by lexical unit and provided frame.
The instruction builder selects only the entries for that frame. `rel` is
normalized to `PRED` at the schema boundary.

The files contain label definitions. Dataset sentences and model weights are
supplied separately. Every evaluated role must have a definition in the supplied
task inventory; conversion and instruction construction report missing roles
instead of removing their examples. Continuation and referent labels retain their
native identities when exported to benchmark scorers.
