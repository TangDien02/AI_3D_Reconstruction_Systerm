from __future__ import annotations

import csv
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ARCH_KEYS = [
    "num_points",
    "image_size",
    "feature_dim",
]
CURRENT_MODEL_TYPE = "resnet_pointcloud"


def project_path(path_text: str) -> Path:
    path = Path(path_text.strip())
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def as_project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path.resolve())


def read_category_counts(csv_path: Path) -> dict[str, int]:
    if not csv_path.is_file():
        return {}

    counts: dict[str, int] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "category" not in (reader.fieldnames or []):
            return {}
        for row in reader:
            category = row.get("category")
            if category:
                counts[category] = counts.get(category, 0) + 1
    return counts


def normalize_categories(categories) -> set[str] | None:
    if categories is None:
        return None
    if isinstance(categories, str):
        return {categories}
    return {str(category) for category in categories}


class TrainingConfigGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Training Config Launcher")
        self.geometry("1040x760")
        self.minsize(920, 680)

        self.process: subprocess.Popen | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        self.metadata_counts = read_category_counts(PROCESSED_DIR / "pix3d_clean_metadata.csv")
        self.train_counts = read_category_counts(PROCESSED_DIR / "splits" / "train.csv")
        self.categories = sorted(self.metadata_counts or self.train_counts or self.folder_categories())

        self.create_variables()
        self.create_widgets()
        self.refresh_defaults_for_category()
        self.after(150, self.drain_output_queue)
        self.validate_config(show_dialog=False)

    def folder_categories(self) -> list[str]:
        image_dir = PROCESSED_DIR / "images"
        if not image_dir.is_dir():
            return []
        return sorted(path.name for path in image_dir.iterdir() if path.is_dir())

    def create_variables(self) -> None:
        default_category = "chair" if "chair" in self.categories else (self.categories[0] if self.categories else "")
        self.category_var = tk.StringVar(value=default_category)
        self.output_dir_var = tk.StringVar(value="results/chair_resnet_baseline")
        self.resume_mode_var = tk.StringVar(value="auto_best")
        self.custom_checkpoint_var = tk.StringVar(value="")
        self.max_samples_var = tk.StringVar(value="")
        self.epochs_var = tk.StringVar(value="5")
        self.batch_size_var = tk.StringVar(value="2")
        self.lr_var = tk.StringVar(value="0.0001")
        self.num_points_var = tk.StringVar(value="2048")
        self.image_size_var = tk.StringVar(value="224")
        self.encoder_name_var = tk.StringVar(value="resnet18")
        self.feature_dim_var = tk.StringVar(value="512")
        self.pretrained_var = tk.BooleanVar(value=True)
        self.freeze_encoder_var = tk.BooleanVar(value=True)
        self.best_metric_var = tk.StringVar(value="val_chamfer_distance")
        self.force_cpu_var = tk.BooleanVar(value=True)

    def create_widgets(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(root, text="Training Config Launcher", font=("Segoe UI", 16, "bold"))
        title.pack(anchor=tk.W)
        desc = ttk.Label(
            root,
            text=(
                "Chon category, output folder va checkpoint. Sau training, pipeline tu sinh metric charts, "
                "epoch table va anh so sanh predicted point cloud voi ground truth."
            ),
        )
        desc.pack(anchor=tk.W, pady=(0, 10))

        body = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=1)

        self.build_config_panel(left)
        self.build_status_panel(right)

    def build_config_panel(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas)
        content_window = canvas.create_window((0, 0), window=content, anchor=tk.NW)

        def update_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def update_content_width(event) -> None:
            canvas.itemconfigure(content_window, width=event.width)

        content.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_content_width)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        self.build_dataset_frame(content)
        self.build_train_frame(content)
        self.build_arch_frame(content)
        self.build_resume_frame(content)

        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        ttk.Button(button_frame, text="Check config", command=lambda: self.validate_config(show_dialog=True)).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        ttk.Button(button_frame, text="Copy command", command=self.copy_command).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_frame, text="Start training", command=self.start_training).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_frame, text="Stop", command=self.stop_training).pack(side=tk.LEFT)

    def build_dataset_frame(self, parent: ttk.Frame) -> None:
        data_frame = ttk.LabelFrame(parent, text="Dataset")
        data_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(data_frame, text="Category").grid(row=0, column=0, sticky=tk.W, padx=8, pady=6)
        self.category_combo = ttk.Combobox(
            data_frame,
            textvariable=self.category_var,
            values=self.categories,
            state="readonly" if self.categories else "normal",
        )
        self.category_combo.grid(row=0, column=1, sticky=tk.EW, padx=8, pady=6)
        self.category_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_category_change())

        ttk.Label(data_frame, text="Output dir").grid(row=1, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(data_frame, textvariable=self.output_dir_var).grid(row=1, column=1, sticky=tk.EW, padx=8, pady=6)
        ttk.Button(data_frame, text="Browse", command=self.choose_output_dir).grid(row=1, column=2, padx=8, pady=6)
        data_frame.columnconfigure(1, weight=1)

    def build_train_frame(self, parent: ttk.Frame) -> None:
        train_frame = ttk.LabelFrame(parent, text="Train parameters")
        train_frame.pack(fill=tk.X, pady=(0, 10))

        fields = [
            ("Max samples", self.max_samples_var),
            ("Epochs to add", self.epochs_var),
            ("Batch size", self.batch_size_var),
            ("Learning rate", self.lr_var),
        ]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(train_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=8, pady=5)
            ttk.Entry(train_frame, textvariable=variable, width=16).grid(row=row, column=1, sticky=tk.W, padx=8, pady=5)

        ttk.Label(train_frame, text="Best metric").grid(row=4, column=0, sticky=tk.W, padx=8, pady=5)
        ttk.Combobox(
            train_frame,
            textvariable=self.best_metric_var,
            values=["val_chamfer_distance", "val_f_score"],
            state="readonly",
            width=24,
        ).grid(row=4, column=1, sticky=tk.W, padx=8, pady=5)

        ttk.Checkbutton(train_frame, text="Force CPU", variable=self.force_cpu_var).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky=tk.W,
            padx=8,
            pady=5,
        )

    def build_arch_frame(self, parent: ttk.Frame) -> None:
        arch_frame = ttk.LabelFrame(parent, text="Model architecture")
        arch_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(arch_frame, text="Encoder").grid(row=0, column=0, sticky=tk.W, padx=8, pady=5)
        encoder_combo = ttk.Combobox(
            arch_frame,
            textvariable=self.encoder_name_var,
            values=["resnet18", "resnet50", "conv"],
            state="readonly",
            width=14,
        )
        encoder_combo.grid(row=0, column=1, sticky=tk.W, padx=8, pady=5)
        encoder_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_encoder_change())

        arch_fields = [
            ("Num points", self.num_points_var),
            ("Image size", self.image_size_var),
            ("Feature dim", self.feature_dim_var),
        ]
        for row, (label, variable) in enumerate(arch_fields):
            grid_row = (row + 2) // 2
            ttk.Label(arch_frame, text=label).grid(row=grid_row, column=((row + 2) % 2) * 2, sticky=tk.W, padx=8, pady=5)
            ttk.Entry(arch_frame, textvariable=variable, width=12).grid(
                row=grid_row,
                column=((row + 2) % 2) * 2 + 1,
                sticky=tk.W,
                padx=8,
                pady=5,
            )
        ttk.Checkbutton(arch_frame, text="Pretrained ImageNet", variable=self.pretrained_var).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky=tk.W,
            padx=8,
            pady=5,
        )
        ttk.Checkbutton(arch_frame, text="Freeze encoder", variable=self.freeze_encoder_var).grid(
            row=3,
            column=2,
            columnspan=2,
            sticky=tk.W,
            padx=8,
            pady=5,
        )

    def build_resume_frame(self, parent: ttk.Frame) -> None:
        resume_frame = ttk.LabelFrame(parent, text="Checkpoint / resume")
        resume_frame.pack(fill=tk.X, pady=(0, 10))

        modes = [
            ("Auto: dung best_model.pt neu co", "auto_best"),
            ("Train model moi", "fresh"),
            ("Bat buoc resume best_model.pt", "best"),
            ("Resume resnet_pointcloud_net.pt", "last"),
            ("Checkpoint tuy chon", "custom"),
        ]
        for row, (label, value) in enumerate(modes):
            ttk.Radiobutton(
                resume_frame,
                text=label,
                variable=self.resume_mode_var,
                value=value,
                command=lambda: self.validate_config(show_dialog=False),
            ).grid(row=row, column=0, sticky=tk.W, padx=8, pady=3)

        custom_row = len(modes)
        ttk.Entry(resume_frame, textvariable=self.custom_checkpoint_var).grid(
            row=custom_row,
            column=0,
            sticky=tk.EW,
            padx=8,
            pady=6,
        )
        ttk.Button(resume_frame, text="Choose checkpoint", command=self.choose_checkpoint).grid(
            row=custom_row,
            column=1,
            padx=8,
            pady=6,
        )
        resume_frame.columnconfigure(0, weight=1)

    def build_status_panel(self, parent: ttk.Frame) -> None:
        status_frame = ttk.LabelFrame(parent, text="Config status")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.status_text = tk.Text(status_frame, height=16, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.status_text.tag_configure("ok", foreground="#166534")
        self.status_text.tag_configure("warn", foreground="#92400e")
        self.status_text.tag_configure("error", foreground="#b91c1c")
        self.status_text.tag_configure("info", foreground="#1d4ed8")

        command_frame = ttk.LabelFrame(parent, text="Command / logs")
        command_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = tk.Text(command_frame, height=18, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def on_category_change(self) -> None:
        self.refresh_defaults_for_category()
        self.validate_config(show_dialog=False)

    def on_encoder_change(self) -> None:
        encoder_defaults = {"resnet18": "512", "resnet50": "2048", "conv": "256"}
        self.feature_dim_var.set(encoder_defaults.get(self.encoder_name_var.get(), "512"))
        self.validate_config(show_dialog=False)

    def refresh_defaults_for_category(self) -> None:
        category = self.category_var.get()
        if not category:
            return
        current = self.output_dir_var.get().strip()
        default_like = {"", "results/baseline", "results/chair_baseline", "results/chair_resnet_baseline"} | {
            f"results/{cat}_baseline" for cat in self.categories
        } | {
            f"results/{cat}_resnet_baseline" for cat in self.categories
        }
        if current in default_like:
            self.output_dir_var.set(f"results/{category}_resnet_baseline")

    def choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=PROJECT_DIR / "results")
        if selected:
            self.output_dir_var.set(as_project_relative(Path(selected)))
            self.validate_config(show_dialog=False)

    def choose_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=PROJECT_DIR / "results",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self.custom_checkpoint_var.set(as_project_relative(Path(selected)))
            self.resume_mode_var.set("custom")
            self.validate_config(show_dialog=False)

    def numeric_config(self) -> tuple[dict[str, int | float | None], list[str]]:
        errors: list[str] = []
        config: dict[str, int | float | None] = {}
        int_fields = {
            "max_samples": self.max_samples_var,
            "epochs": self.epochs_var,
            "batch_size": self.batch_size_var,
            "num_points": self.num_points_var,
            "image_size": self.image_size_var,
            "feature_dim": self.feature_dim_var,
        }
        for key, variable in int_fields.items():
            text = variable.get().strip()
            if key == "max_samples" and text == "":
                config[key] = None
                continue
            try:
                value = int(text)
            except ValueError:
                errors.append(f"{key} phai la so nguyen.")
                continue
            if value <= 0:
                errors.append(f"{key} phai lon hon 0.")
            config[key] = value

        try:
            lr = float(self.lr_var.get().strip())
            if lr <= 0:
                errors.append("learning rate phai lon hon 0.")
            config["lr"] = lr
        except ValueError:
            errors.append("learning rate phai la so.")
        return config, errors

    def resolve_checkpoint_path(self) -> tuple[Path | None, str]:
        output_dir = project_path(self.output_dir_var.get())
        mode = self.resume_mode_var.get()
        if mode == "fresh":
            return None, "fresh"
        if mode in {"auto_best", "best"}:
            return output_dir / "outputs" / "checkpoints" / "best_model.pt", mode
        if mode == "last":
            return output_dir / "outputs" / "checkpoints" / "resnet_pointcloud_net.pt", mode
        return project_path(self.custom_checkpoint_var.get()), mode

    def load_checkpoint(self, checkpoint_path: Path) -> tuple[dict | None, str | None]:
        try:
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if not isinstance(checkpoint, dict):
                return None, "Checkpoint khong phai dict PyTorch hop le."
            return checkpoint, None
        except Exception as exc:
            return None, f"Khong doc duoc checkpoint: {exc}"

    def validate_config(self, show_dialog: bool = False) -> tuple[bool, list[str], list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        infos: list[str] = []
        category = self.category_var.get().strip()
        output_dir = project_path(self.output_dir_var.get())
        numeric, numeric_errors = self.numeric_config()
        errors.extend(numeric_errors)

        if not category:
            errors.append("Chua chon category.")
        elif category not in self.metadata_counts and self.metadata_counts:
            errors.append(f"Category '{category}' khong co trong pix3d_clean_metadata.csv.")

        if not (PROCESSED_DIR / "splits" / "train.csv").is_file():
            errors.append("Thieu data/processed/splits/train.csv. Hay preprocessing truoc.")
        else:
            train_count = self.train_counts.get(category, 0)
            if train_count <= 1:
                errors.append(f"Train split cua category '{category}' qua it mau: {train_count}.")
            else:
                max_samples = numeric.get("max_samples")
                effective = min(train_count, int(max_samples)) if max_samples else train_count
                if effective <= 1:
                    errors.append("max_samples qua nho, can it nhat 2 mau de train/val.")
                infos.append(f"Train samples available for {category}: {train_count}; effective max: {effective}.")

        checkpoint_path, mode = self.resolve_checkpoint_path()
        if mode == "fresh":
            existing_best = output_dir / "outputs" / "checkpoints" / "best_model.pt"
            if existing_best.is_file():
                warnings.append(
                    "Dang chon train model moi trong output dir da co best_model.pt; checkpoint cu co the bi ghi de."
                )
            infos.append("Resume disabled: train tu dau.")
        elif mode == "auto_best" and checkpoint_path and not checkpoint_path.is_file():
            infos.append("Chua co best_model.pt trong output dir; lan nay se train model moi.")
        else:
            if not checkpoint_path or not checkpoint_path.is_file():
                errors.append(f"Khong tim thay checkpoint: {checkpoint_path}")

        if checkpoint_path and checkpoint_path.is_file():
            checkpoint, checkpoint_error = self.load_checkpoint(checkpoint_path)
            if checkpoint_error:
                errors.append(checkpoint_error)
            elif checkpoint:
                infos.append(f"Checkpoint: {as_project_relative(checkpoint_path)}")
                infos.append(f"Checkpoint epoch: {checkpoint.get('epoch')}; best_score: {checkpoint.get('best_score')}")
                checkpoint_model_type = checkpoint.get("model_type")
                if checkpoint_model_type not in {None, CURRENT_MODEL_TYPE}:
                    errors.append(
                        f"Sai model_type: checkpoint={checkpoint_model_type} current={CURRENT_MODEL_TYPE}."
                    )
                if checkpoint_model_type is None and any(
                    key in checkpoint for key in ("patch_size", "embed_dim", "transformer_depth", "num_heads")
                ):
                    errors.append("Checkpoint nay la Transformer cu; hay train ResNet checkpoint moi.")
                checkpoint_categories = normalize_categories(checkpoint.get("categories"))
                current_categories = {category}
                if checkpoint_categories is not None and checkpoint_categories != current_categories:
                    errors.append(
                        f"Sai category: checkpoint={sorted(checkpoint_categories)} current={sorted(current_categories)}."
                    )
                if checkpoint.get("encoder_name") is not None and checkpoint.get("encoder_name") != self.encoder_name_var.get():
                    errors.append(
                        f"Sai encoder_name: checkpoint={checkpoint.get('encoder_name')} current={self.encoder_name_var.get()}."
                    )
                for key, variable in {
                    "pretrained": self.pretrained_var,
                    "freeze_encoder": self.freeze_encoder_var,
                }.items():
                    if checkpoint.get(key) is not None and bool(checkpoint.get(key)) != bool(variable.get()):
                        errors.append(f"Sai {key}: checkpoint={checkpoint.get(key)} current={variable.get()}.")
                for key in ARCH_KEYS:
                    current_value = numeric.get(key)
                    checkpoint_value = checkpoint.get(key)
                    if checkpoint_value is not None and current_value is not None:
                        if int(checkpoint_value) != int(current_value):
                            errors.append(f"Sai {key}: checkpoint={checkpoint_value} current={current_value}.")

        infos.append("Command:")
        infos.append(" ".join(self.build_command()))
        infos.append("Auto outputs after training:")
        infos.append("metrics/training_metrics.csv, metrics/test_summary.json, metrics/test_batch_metrics.csv")
        infos.append("outputs/training_curves.png, outputs/test_summary_metrics.png, outputs/test_batch_metrics.png")
        infos.append("outputs/comparison/<sample_id>_comparison.png")

        self.write_status(errors, warnings, infos)
        if show_dialog:
            if errors:
                messagebox.showerror("Config invalid", "\n".join(errors[:8]))
            elif warnings:
                messagebox.showwarning("Config warning", "\n".join(warnings[:8]))
            else:
                messagebox.showinfo("Config OK", "Cau hinh hop le, co the train.")
        return not errors, errors, warnings, infos

    def write_status(self, errors: list[str], warnings: list[str], infos: list[str]) -> None:
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        if errors:
            self.status_text.insert(tk.END, "ERROR\n", "error")
            for item in errors:
                self.status_text.insert(tk.END, f"- {item}\n", "error")
            self.status_text.insert(tk.END, "\n")
        if warnings:
            self.status_text.insert(tk.END, "WARNING\n", "warn")
            for item in warnings:
                self.status_text.insert(tk.END, f"- {item}\n", "warn")
            self.status_text.insert(tk.END, "\n")
        if not errors:
            self.status_text.insert(tk.END, "OK\n", "ok")
            self.status_text.insert(tk.END, "- Config co the chay.\n\n", "ok")
        self.status_text.insert(tk.END, "INFO\n", "info")
        for item in infos:
            self.status_text.insert(tk.END, f"- {item}\n", "info")
        self.status_text.configure(state=tk.DISABLED)

    def selected_device_name(self) -> str:
        return "cpu" if self.force_cpu_var.get() else "auto"

    def build_command(self) -> list[str]:
        checkpoint_path, mode = self.resolve_checkpoint_path()
        command = [
            sys.executable,
            "-m",
            "src.training.training_pipeline",
            "--dataset-mode",
            "processed",
            "--categories",
            self.category_var.get().strip(),
            "--epochs",
            self.epochs_var.get().strip(),
            "--batch-size",
            self.batch_size_var.get().strip(),
            "--lr",
            self.lr_var.get().strip(),
            "--output-dir",
            self.output_dir_var.get().strip(),
            "--num-points",
            self.num_points_var.get().strip(),
            "--image-size",
            self.image_size_var.get().strip(),
            "--encoder-name",
            self.encoder_name_var.get().strip(),
            "--feature-dim",
            self.feature_dim_var.get().strip(),
            "--device",
            self.selected_device_name(),
            "--best-metric",
            self.best_metric_var.get().strip(),
        ]
        command.append("--pretrained" if self.pretrained_var.get() else "--no-pretrained")
        command.append("--freeze-encoder" if self.freeze_encoder_var.get() else "--no-freeze-encoder")
        if self.max_samples_var.get().strip():
            command.extend(["--max-samples", self.max_samples_var.get().strip()])
        if mode == "fresh":
            command.append("--no-resume")
        elif mode in {"best", "last", "custom"} and checkpoint_path is not None:
            command.extend(["--resume-checkpoint", as_project_relative(checkpoint_path)])
        return command

    def copy_command(self) -> None:
        command = " ".join(self.build_command())
        self.clipboard_clear()
        self.clipboard_append(command)
        messagebox.showinfo("Copied", "Da copy command vao clipboard.")

    def start_training(self) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Training running", "Training dang chay.")
            return

        valid, errors, warnings, _infos = self.validate_config(show_dialog=False)
        if not valid:
            messagebox.showerror("Config invalid", "\n".join(errors[:8]))
            return
        if warnings:
            proceed = messagebox.askyesno("Config warning", "\n".join(warnings[:8]) + "\n\nVan tiep tuc train?")
            if not proceed:
                return

        command = self.build_command()
        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        if self.force_cpu_var.get():
            env["CUDA_VISIBLE_DEVICES"] = ""

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Starting training...\n")
        self.output_text.insert(tk.END, " ".join(command) + "\n\n")

        thread = threading.Thread(target=self.run_process, args=(command, env), daemon=True)
        thread.start()

    def run_process(self, command: list[str], env: dict[str, str]) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(line)
            code = self.process.wait()
            self.output_queue.put(f"\nProcess finished with exit code {code}.\n")
        except Exception as exc:
            self.output_queue.put(f"\nFailed to start training: {exc}\n")
        finally:
            self.process = None

    def drain_output_queue(self) -> None:
        while True:
            try:
                text = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)
        self.after(150, self.drain_output_queue)

    def stop_training(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.output_text.insert(tk.END, "\nTerminate requested.\n")
        else:
            messagebox.showinfo("No process", "Khong co training process dang chay.")


def main() -> None:
    app = TrainingConfigGui()
    app.mainloop()


if __name__ == "__main__":
    main()
