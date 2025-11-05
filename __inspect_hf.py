import inspect
import huggingface_hub.hub_mixin as hm

print(dir(hm))

mixins = [getattr(hm, name) for name in dir(hm) if name.endswith("Mixin")]
for mixin in mixins:
	if hasattr(mixin, "from_pretrained"):
		sig = inspect.signature(mixin.from_pretrained)
		print(mixin.__name__, sig)
