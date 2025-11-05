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
		print(f"Found class {node.name}")
		if node.name == "SALM":
			bases = [ast.unparse(base) for base in node.bases]
			print("Bases:", bases)
			for item in node.body:
				if isinstance(item, ast.FunctionDef) and item.name == "from_pretrained":
					args = [arg.arg for arg in item.args.args]
					kwonly = [arg.arg for arg in item.args.kwonlyargs]
					print("args:", args)
					print("kwonly:", kwonly)
					defaults = len(item.args.defaults)
					kw_defaults = [None if d is None else ast.unparse(d) for d in item.args.kw_defaults]
					print("kw_defaults:", kw_defaults)

	def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
		if node.name == "from_pretrained":
			print("Top-level from_pretrained found")
		print(f"Function {node.name}")
		args = [arg.arg for arg in node.args.args]
		kwonly = [arg.arg for arg in node.args.kwonlyargs]
		print("args:", args)
		print("kwonly:", kwonly)
		kw_defaults = [None if d is None else ast.unparse(d) for d in node.args.kw_defaults]
		print("kw_defaults:", kw_defaults)

FuncVisitor().visit(module)
