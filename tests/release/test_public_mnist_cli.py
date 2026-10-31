"""Public CLI regression checks without neural imports, downloads or training."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import mnist


def args(*words):
    return mnist.parser().parse_args(words)


class PublicHelpTests(unittest.TestCase):
    def test_help_uses_only_neutral_names_without_neural_dependencies(self):
        commands = ((), ("list-models",), ("verify",), ("checkpoints",), ("evaluate",),
                    ("profile",), ("train",), ("prepare-classifier",))
        environment = dict(os.environ)
        environment.pop("PYTHONPATH",None)
        environment["PYTHONIOENCODING"] = "utf-8"
        with tempfile.TemporaryDirectory(prefix="fm-cli-help-") as folder:
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable,str(ROOT/"mnist.py"),*command,"--help"],
                                            cwd=folder,env=environment,capture_output=True,text=True,timeout=20)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertNotIn("iclr",(result.stdout+result.stderr).lower())
                    self.assertIn("mnist.py",result.stdout)

    def test_module_import_is_standard_library_only(self):
        result = subprocess.run([sys.executable,"-c",
            "import mnist,sys; assert not any(n in sys.modules for n in ('torch','torchvision','compressai')); print('PASS')"],
            cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_model_list_is_path_free_and_has_four_operating_points(self):
        with tempfile.TemporaryDirectory(prefix="fm-cli-models-") as folder:
            root = Path(folder).resolve()
            for relative in (*mnist.MODEL_PATHS.values(),mnist.CLASSIFIER_PATH):
                path = root/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b"synthetic model bytes, never loaded")
            inventory = mnist.model_inventory(root)
            self.assertEqual([row["rate_weight"] for row in inventory["codecs"]],list(mnist.RATES))
            self.assertTrue(all(row["available"] for row in inventory["codecs"]))
            self.assertTrue(inventory["classifier"]["available"])
            self.assertNotIn("iclr",json.dumps(inventory).lower())
            self.assertNotIn(str(root),json.dumps(inventory))

    def test_actual_bundled_models_are_listed_without_neural_execution(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(mnist.main(["list-models","--json"]),0)
        inventory = json.loads(output.getvalue())
        self.assertEqual(len(inventory["codecs"]),4)
        self.assertNotIn("iclr",output.getvalue().lower())


class InvocationTests(unittest.TestCase):
    def test_missing_optional_dependencies_stop_before_engine_execution(self):
        errors = io.StringIO()
        with patch.object(mnist,"require_neural_dependencies",side_effect=RuntimeError("Optional neural dependencies are missing")), \
                patch.object(mnist.subprocess,"run") as launch, contextlib.redirect_stderr(errors):
            self.assertEqual(mnist.main(["evaluate","--rate-weight","5000"]),2)
        launch.assert_not_called()
        self.assertNotIn("iclr",errors.getvalue().lower())

    def test_evaluate_uses_published_escape_protocol(self):
        command = mnist.command_for(args("evaluate","--rate-weight","5000"))
        for flag,value in (("--symbol-minimum","-128"),("--symbol-maximum","128"),
                           ("--precision","16"),("--maximum-samples","5000"),
                           ("--schedule-repetition","0")):
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertIn("--escape-tail",command)
        self.assertEqual(Path(command[command.index("--output")+1]),ROOT/"outputs/mnist/rate_5000_rep_0.json")

    def test_evaluate_keeps_requested_repetition_and_identity_control(self):
        command = mnist.command_for(args("evaluate","--rate-weight","200","--schedule-repetition","4",
                                           "--device","cuda:0","--levels","1","31"))
        self.assertEqual(command[command.index("--schedule-repetition")+1],"4")
        self.assertEqual(command[command.index("--device")+1],"cuda:0")
        self.assertEqual(command[command.index("--levels")+1:command.index("--maximum-samples")],["1","31"])
        with self.assertRaises(ValueError):
            mnist.command_for(args("evaluate","--rate-weight","200","--levels","3","31"))
        with self.assertRaises(ValueError):
            mnist.command_for(args("evaluate","--rate-weight","200","--levels","1","1"))

    def test_profile_uses_neutral_output_and_recorded_defaults(self):
        command = mnist.command_for(args("profile","--rate-weight","50000"))
        self.assertEqual(command[command.index("--output-dir")+1],str(ROOT/"outputs/edge_profile"))
        self.assertEqual(command[command.index("--threads")+1],"1")
        self.assertEqual(command[command.index("--samples")+1],"128")
        self.assertEqual(command[command.index("--repetitions")+1],"100")
        self.assertEqual(command[command.index("--warmup")+1],"20")

    def test_original_source_and_model_directories_are_protected(self):
        for relative in (".","data/report.json","results/new.json","configs/new.json",
                         "validation/new.json","neural/new.json",".git/new",".github/workflows/new.json",
                         "arbitrary-source-dir/new.json"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                mnist.command_for(args("evaluate","--rate-weight","0","--output",relative))

    def test_checkpoints_and_classifier_use_safe_working_outputs(self):
        command = mnist.command_for(args("checkpoints","--run-cpu-smoke"))
        self.assertIn("--run-cpu-smoke",command)
        self.assertEqual(command[command.index("--output")+1],str(ROOT/"outputs/checkpoint_validation.json"))
        command = mnist.command_for(args("prepare-classifier"))
        self.assertEqual(command[command.index("--output")+1],str(ROOT/"runs/classifier/mnist_classifier.pt"))
        with self.assertRaises(ValueError):
            mnist.command_for(args("prepare-classifier","--output","results/classifier.pt"))


class NewTrainingTests(unittest.TestCase):
    def make_grid(self, root):
        payload = json.loads((ROOT/mnist.GRID).read_text(encoding="utf-8"))
        path = root/"source-grid.json"
        path.write_text(json.dumps(payload),encoding="utf-8")
        return path,payload

    def test_training_plan_does_not_modify_frozen_grid_or_values(self):
        with tempfile.TemporaryDirectory(prefix="fm-cli-training-") as folder:
            root = Path(folder).resolve()
            grid,payload = self.make_grid(root)
            original = grid.read_bytes()
            tasks = mnist.training_plan(args("train","--grid",str(grid)),root)
            self.assertEqual(len(tasks),4)
            self.assertEqual(grid.read_bytes(),original)
            for config_path,config,action in tasks:
                self.assertTrue(config_path.is_relative_to(root/"runs/mnist_training/configs"))
                self.assertTrue(Path(config["output_dir"]).is_relative_to(root/"runs/mnist_training"))
                self.assertEqual(action,"train-evaluate")
                expected = dict(payload["base_config"]["train"],rate_weight=config["train"]["rate_weight"])
                self.assertEqual(config["train"],expected)
                self.assertEqual(config["model"],payload["base_config"]["model"])
            self.assertFalse((root/"results").exists())

    def test_training_operating_point_and_shards(self):
        with tempfile.TemporaryDirectory(prefix="fm-cli-training-") as folder:
            root = Path(folder).resolve()
            grid,_ = self.make_grid(root)
            one = mnist.training_plan(args("train","--grid",str(grid),"--rate-weight","5000"),root)
            self.assertEqual(len(one),1)
            self.assertEqual(one[0][1]["train"]["rate_weight"],5000)
            shards = [mnist.training_plan(args("train","--grid",str(grid),"--shard-count","4","--shard-index",str(n)),root) for n in range(4)]
            self.assertEqual([len(shard) for shard in shards],[1,1,1,1])
            self.assertEqual(len({item[0] for shard in shards for item in shard}),4)
            with self.assertRaises(ValueError):
                mnist.training_plan(args("train","--grid",str(grid),"--shard-count","1","--shard-index","1"),root)

    def test_training_never_overwrites_existing_new_run_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="fm-cli-training-") as folder:
            root = Path(folder).resolve()
            grid,_ = self.make_grid(root)
            options = args("train","--grid",str(grid),"--rate-weight","0")
            task = mnist.training_plan(options,root)[0]
            checkpoint = Path(task[1]["output_dir"])/"final.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                mnist.training_plan(options,root)
            resumed = mnist.training_plan(args("train","--grid",str(grid),"--rate-weight","0","--resume"),root)
            self.assertEqual(resumed[0][2],"evaluate")
            self.assertEqual(checkpoint.read_bytes(),b"existing")

    def test_training_dispatch_is_separate_and_deterministic(self):
        with tempfile.TemporaryDirectory(prefix="fm-cli-training-") as folder:
            root = Path(folder).resolve()
            grid,_ = self.make_grid(root)
            options = args("train","--grid",str(grid),"--rate-weight","200")
            with patch.object(mnist.subprocess,"run",return_value=SimpleNamespace(returncode=0)) as launch:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(mnist.run_training(options,root),0)
                launch.assert_called_once()
                command = launch.call_args.args[0]
                self.assertEqual(command[command.index("--action")+1],"train-evaluate")
                self.assertEqual(launch.call_args.kwargs["cwd"],root)
                config_path = Path(command[command.index("--config")+1])
                self.assertTrue(config_path.is_relative_to(root/"runs"))
                self.assertTrue(config_path.is_file())


if __name__ == "__main__":
    unittest.main()
