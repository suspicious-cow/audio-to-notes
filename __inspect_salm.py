import ast
from pathlib import Path

env_root = Path("C:/Users/Zain_/anaconda3/envs/audio-notes-gpu/Lib/site-packages")
models_dir = env_root / "nemo" / "collections" / "speechlm2" / "models"
print("Models dir contents:")
for child in models_dir.iterdir():
	print("-", child.name)

target = models_dir / "salm.py"

print(f"Reading {target}")
source = target.read_text(encoding="utf-8")
module = ast.parse(source)

class FuncVisitor(ast.NodeVisitor):
	def visit_ClassDef(self, node: ast.ClassDef) -> None:
		if node.name == "SALM":
			print(f"Found class {node.name}")
			bases = [ast.unparse(base) for base in node.bases]
			print("Bases:", bases)
			for item in node.body:
				if isinstance(item, ast.FunctionDef) and item.name == "__init__":
					args = [arg.arg for arg in item.args.args]
					kwonly = [arg.arg for arg in item.args.kwonlyargs]
					defaults = [None if d is None else ast.unparse(d) for d in item.args.defaults]
					kw_defaults = [None if d is None else ast.unparse(d) for d in item.args.kw_defaults]
					print("__init__ args:", args)
					print("__init__ kwonly:", kwonly)
					print("__init__ defaults:", defaults)
					print("__init__ kw_defaults:", kw_defaults)
					segment = ast.get_source_segment(source, item)
					if segment:
						print("__init__ snippet:\n" + '\n'.join(segment.split('\n')[:80]))

	def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
		pass


FuncVisitor().visit(module)
