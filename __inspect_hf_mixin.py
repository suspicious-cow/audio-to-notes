import ast
from pathlib import Path

env_root = Path("C:/Users/Zain_/anaconda3/envs/audio-notes-gpu/Lib/site-packages")
parts_dir = env_root / "nemo" / "collections" / "speechlm2" / "parts"
target = parts_dir / "hf_hub.py"
print(f"Reading {target}")
source = target.read_text(encoding="utf-8")
first_lines = source.splitlines()[:20]
print("First 20 lines preview:")
for line in first_lines:
    print(line)
module = ast.parse(source)


class Visitor(ast.NodeVisitor):
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == "HFHubMixin":
            print(f"Found class {node.name}")
            bases = [ast.unparse(base) for base in node.bases]
            print("Bases:", bases)
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "from_pretrained":
                    args = [arg.arg for arg in item.args.args]
                    kwonly = [arg.arg for arg in item.args.kwonlyargs]
                    defaults = [None if d is None else ast.unparse(d) for d in item.args.defaults]
                    kw_defaults = [None if d is None else ast.unparse(d) for d in item.args.kw_defaults]
                    print("args:", args)
                    print("kwonly:", kwonly)
                    print("defaults:", defaults)
                    print("kw_defaults:", kw_defaults)
                if isinstance(item, ast.FunctionDef) and item.name == "_from_pretrained":
                    args = [arg.arg for arg in item.args.args]
                    kwonly = [arg.arg for arg in item.args.kwonlyargs]
                    defaults = [None if d is None else ast.unparse(d) for d in item.args.defaults]
                    kw_defaults = [None if d is None else ast.unparse(d) for d in item.args.kw_defaults]
                    print("_from_pretrained args:", args)
                    print("_from_pretrained kwonly:", kwonly)
                    print("_from_pretrained defaults:", defaults)
                    print("_from_pretrained kw_defaults:", kw_defaults)
                    segment = ast.get_source_segment(source, item)
                    if segment:
                        print("_from_pretrained source:\n" + segment)
        methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
        print("Methods:", methods)

        # Continue traversal to ensure nested classes analysed
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "from_pretrained":
            args = [arg.arg for arg in node.args.args]
            kwonly = [arg.arg for arg in node.args.kwonlyargs]
            defaults = [None if d is None else ast.unparse(d) for d in node.args.defaults]
            kw_defaults = [None if d is None else ast.unparse(d) for d in node.args.kw_defaults]
            print(f"Top-level function {node.name}")
            print("args:", args)
            print("kwonly:", kwonly)
            print("defaults:", defaults)
            print("kw_defaults:", kw_defaults)


Visitor().visit(module)
