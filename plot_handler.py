import numpy as np
import matplotlib.pyplot as plt
import os
import base64
import csv
import io

from control_panel import ControlPanel
from openai import OpenAI

controller = ControlPanel()

class PlotHandler:
    def plot_for_paper(self):
        # self.plot_sft_loss()
        # self.plot_sft_accuracy()
        # self.plot_sft_loss_and_accuracy()
        # self.plot_sft_and_dpo_loss_and_accuracy()
        # self.plot_dpo_loss() # this doesn't work because of the API problems
        # self.plot_dpo_loss_2()
        # self.plot_reconstruction_error()
        self.plot_moca_score_error()
        pass

    def plot_cvae_losses(self, loss_matrix, save_dir, file_name, labels=None, normalize=True, fig_size=(7, 4)):
        loss_matrix = np.asarray(loss_matrix, dtype=float)
        n_losses, n_points = loss_matrix.shape

        if labels is None:
            labels = [f"Loss {i + 1}" for i in range(n_losses)]

        # Normalize each loss vector independently
        if normalize:
            normalized_losses = []
            for loss in loss_matrix:
                min_val = np.min(loss)
                max_val = np.max(loss)

                if max_val == min_val:
                    normalized_loss = np.zeros_like(loss)
                else:
                    normalized_loss = (loss - min_val) / (max_val - min_val)

                normalized_losses.append(normalized_loss)

            loss_matrix = np.array(normalized_losses)

        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, file_name)

        colors = ["r", "g", "b", "m"]  # 4 colors
        x = np.arange(n_points)

        plt.rcParams.update({"font.size": 14})
        plt.figure(figsize=fig_size)

        for i in range(n_losses):
            plt.plot(x, loss_matrix[i], label=labels[i], linewidth=1, color=colors[i % len(colors)], linestyle="-")

        plt.xlabel("")
        plt.ylabel("")
        plt.title("")
        plt.legend()

        x_grid_step = n_points // 5

        if n_points > 0:
            plt.xticks(list(range(0, n_points + x_grid_step, x_grid_step)))
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.margins(x=0)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Horizontal grid lines
        for yv in ax.get_yticks():
            ax.axhline(y=yv, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)

        # Vertical grid lines
        x_min, x_max = ax.get_xlim()
        for xv in np.arange(x_min, x_max + x_grid_step, x_grid_step):
            ax.axvline(x=xv, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)

        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


    def plot_sft_loss(self):
        with open(controller.api_key_path, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        job_id = 'ftjob-9C0ymtGC6yHVhZzUaBkDVy0g'
        job = client.fine_tuning.jobs.retrieve(job_id)
        result_file_id = job.result_files[0]
        content = client.files.content(result_file_id)
        raw_text = content.text
        csv_text = base64.b64decode(raw_text).decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_text))
        train_losses = [float(row["train_loss"]) for row in reader if row.get("train_loss")]
        selected_losses = train_losses

        plot_path = controller.plot_path
        os.makedirs(plot_path, exist_ok=True)
        plot_path = os.path.join(controller.plot_path, 'sft_loss.svg')

        ranges = range(0, len(selected_losses))
        plt.rcParams.update({"font.size": 14})
        plt.figure(figsize=(15, 4))
        plt.plot(ranges, selected_losses, linewidth=1, color='r', linestyle='-')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')
        plt.xticks(list(np.arange(0, max(ranges)+1, 100)))  # remove x tick labels
        plt.yticks(range(1, int(max(selected_losses))+2))
        plt.margins(x=0)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_vals = ax.get_yticks()
        for yv in y_vals:
            ax.axhline(y=yv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        x_min, x_max = ax.get_xlim()
        for xv in np.arange(x_min, x_max + 1, 50):
            ax.axvline(x=xv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        plt.savefig(plot_path, bbox_inches='tight')

    def plot_sft_accuracy(self):
        with open(controller.api_key_path, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        job_id = 'ftjob-9C0ymtGC6yHVhZzUaBkDVy0g'
        job = client.fine_tuning.jobs.retrieve(job_id)
        result_file_id = job.result_files[0]
        content = client.files.content(result_file_id)
        raw_text = content.text
        csv_text = base64.b64decode(raw_text).decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_text))
        train_accuracy = [float(row["train_accuracy"]) for row in reader if row.get("train_accuracy")]
        selected_accuracy = train_accuracy

        plot_path = controller.plot_path
        os.makedirs(plot_path, exist_ok=True)
        plot_path = os.path.join(controller.plot_path, 'sft_accuracy.svg')

        ranges = range(0, len(selected_accuracy))
        plt.rcParams.update({"font.size": 14})
        plt.figure(figsize=(15, 4))
        plt.plot(ranges, selected_accuracy, linewidth=1, color='g', linestyle='-')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')
        plt.xticks(list(np.arange(0, max(ranges)+1, 100)))  # remove x tick labels
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.margins(x=0)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_vals = ax.get_yticks()
        for yv in y_vals:
            ax.axhline(y=yv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        x_min, x_max = ax.get_xlim()
        for xv in np.arange(x_min, x_max + 1, 50):
            ax.axvline(x=xv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        plt.savefig(plot_path, bbox_inches='tight')

    def plot_sft_loss_and_accuracy(self):
        with open(controller.api_key_path, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        job_id = 'ftjob-9C0ymtGC6yHVhZzUaBkDVy0g'
        job = client.fine_tuning.jobs.retrieve(job_id)
        result_file_id = job.result_files[0]
        content = client.files.content(result_file_id)
        raw_text = content.text
        csv_text = base64.b64decode(raw_text).decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_text))
        train_loss = np.array([float(row["train_loss"]) for row in reader if row.get("train_loss")])

        reader = csv.DictReader(io.StringIO(csv_text))
        train_accuracy = np.array([float(row["train_accuracy"]) for row in reader if row.get("train_accuracy")])

        # train_loss = train_loss / max(train_loss)
        # train_accuracy = train_accuracy / max(train_accuracy)

        train_loss = (train_loss-min(train_loss)) / (max(train_loss)-min(train_loss))
        train_accuracy = (train_accuracy-min(train_accuracy)) / (max(train_accuracy)-min(train_accuracy))

        plot_path = controller.plot_path
        os.makedirs(plot_path, exist_ok=True)
        plot_path = os.path.join(controller.plot_path, 'sft_loss_and_accuracy.svg')

        ranges = range(0, len(train_loss))
        plt.rcParams.update({"font.size": 14})
        plt.figure(figsize=(15, 4))
        plt.plot(ranges, train_loss, label="Loss", linewidth=1, color='r', linestyle='-')
        plt.plot(ranges, train_accuracy, label="Accuracy", linewidth=1, color='g', linestyle='-')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')
        plt.legend()
        plt.xticks(list(np.arange(0, max(ranges) + 1, 100)))  # remove x tick labels
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.margins(x=0)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_vals = ax.get_yticks()
        for yv in y_vals:
            ax.axhline(y=yv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        x_min, x_max = ax.get_xlim()
        for xv in np.arange(x_min, x_max + 1, 50):
            ax.axvline(x=xv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        plt.savefig(plot_path, bbox_inches='tight')

    def plot_sft_and_dpo_loss_and_accuracy(self):
        with open(controller.api_key_path, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        job_id = 'ftjob-9C0ymtGC6yHVhZzUaBkDVy0g'
        job = client.fine_tuning.jobs.retrieve(job_id)
        result_file_id = job.result_files[0]
        content = client.files.content(result_file_id)
        raw_text = content.text
        csv_text = base64.b64decode(raw_text).decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_text))
        sft_train_loss = np.array([float(row["train_loss"]) for row in reader if row.get("train_loss")])
        reader = csv.DictReader(io.StringIO(csv_text))
        train_accuracy = np.array([float(row["train_accuracy"]) for row in reader if row.get("train_accuracy")])

        dpo_train_loss = []
        with open(os.path.join(controller.result_path, 'train_loss_for_dpo.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dpo_train_loss.append(float(row["Training loss"]))  # convert to float if needed
        dpo_train_loss = np.array(dpo_train_loss[::-1][0:len(sft_train_loss)])

        sft_train_loss = (sft_train_loss-min(sft_train_loss)) / (max(sft_train_loss)-min(sft_train_loss))
        dpo_train_loss = (dpo_train_loss-min(dpo_train_loss)) / (max(dpo_train_loss)-min(dpo_train_loss))
        train_accuracy = (train_accuracy-min(train_accuracy)) / (max(train_accuracy)-min(train_accuracy))

        plot_path = controller.plot_path
        os.makedirs(plot_path, exist_ok=True)
        plot_path = os.path.join(controller.plot_path, 'sft_and_dpo_loss_and_accuracy.pdf')

        ranges = range(0, len(sft_train_loss))
        plt.rcParams.update({"font.size": 17})
        plt.figure(figsize=(15, 4))
        plt.plot(ranges, sft_train_loss, label="SFT Loss", linewidth=1, color='r', linestyle='-')
        plt.plot(ranges, dpo_train_loss, label="DPO Loss", linewidth=1, color='b', linestyle='-')
        plt.plot(ranges, train_accuracy, label="Accuracy", linewidth=1, color='g', linestyle='-')
        plt.xlabel('Batch Number')
        plt.ylabel('Normalized Value')
        plt.title('')
        plt.legend()
        plt.xticks(list(np.arange(0, max(ranges) + 1, 100)))  # remove x tick labels
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.margins(x=0)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_vals = ax.get_yticks()
        for yv in y_vals:
            ax.axhline(y=yv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        x_min, x_max = ax.get_xlim()
        for xv in np.arange(x_min, x_max + 1, 50):
            ax.axvline(x=xv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        plt.savefig(plot_path, bbox_inches='tight')
    
    def plot_dpo_loss(self):
        with open(controller.api_key_path, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        job_id = 'ftjob-XFCKc5OvRvoNf1vVkrUY3jih'
        job = client.fine_tuning.jobs.retrieve(job_id)
        result_file_id = job.result_files[0]
        content = client.files.content(result_file_id)
        raw_text = content.text
        csv_text = base64.b64decode(raw_text).decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        print(rows[0].keys())
        train_losses = [float(row["train_loss"]) for row in reader if row.get("train_loss")]
        selected_losses = train_losses

        plot_path = controller.plot_path
        os.makedirs(plot_path, exist_ok=True)
        plot_path = os.path.join(controller.plot_path, 'dpo_loss.svg')

        ranges = range(0, len(selected_losses))
        plt.rcParams.update({"font.size": 14})
        plt.figure(figsize=(15, 4))
        plt.plot(ranges, selected_losses, linewidth=1, color='g', linestyle='-')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')
        plt.xticks(np.arange(0, max(ranges)+1, 100))
        plt.yticks(np.arange(0, max(selected_losses), 1))
        plt.margins(x=0)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_vals = ax.get_yticks()
        for yv in y_vals:
            ax.axhline(y=yv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        x_min, x_max = ax.get_xlim()
        for xv in np.arange(x_min, x_max + 1, 50):
            ax.axvline(x=xv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        plt.savefig(plot_path, bbox_inches='tight')

    def plot_dpo_loss_2(self):
        train_losses = []
        with open(os.path.join(controller.result_path, 'train_loss_for_dpo.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                train_losses.append(float(row["Training loss"]))  # convert to float if needed

        # Reverse list
        train_losses = train_losses[::-1]
        selected_losses = train_losses

        plot_path = controller.plot_path
        os.makedirs(plot_path, exist_ok=True)
        plot_path = os.path.join(controller.plot_path, 'dpo_loss.svg')

        ranges = range(0, len(selected_losses))
        plt.rcParams.update({"font.size": 14})
        plt.figure(figsize=(15, 4))
        plt.plot(ranges, selected_losses, linewidth=1, color='g', linestyle='-')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')
        plt.xticks(np.arange(0, max(ranges)+1, 100))
        plt.yticks(np.arange(0, max(selected_losses), 1))
        plt.margins(x=0)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_vals = ax.get_yticks()
        for yv in y_vals:
            ax.axhline(y=yv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        x_min, x_max = ax.get_xlim()
        for xv in np.arange(x_min, x_max + 1, 50):
            ax.axvline(x=xv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
        plt.savefig(plot_path, bbox_inches='tight')

    def plot_audio_mimicker_loss(self, data):
        plot_path = controller.plot_path
        os.makedirs(plot_path, exist_ok=True)
        plot_path = os.path.join(controller.plot_path, 'audio_mimicker_loss.svg')

        # Ensure data is clean float list
        selected_losses = [float(x) for x in data if x is not None]

        ranges = range(0, len(selected_losses))

        plt.rcParams.update({"font.size": 14})
        plt.figure(figsize=(15, 4))

        plt.plot(
            ranges,
            selected_losses,
            linewidth=1,
            color='r',
            linestyle='-'
        )

        plt.xlabel('')
        plt.ylabel('')
        plt.title('')

        if len(selected_losses) > 0:
            plt.xticks(list(np.arange(0, max(ranges) + 1, 100)))
            plt.yticks(range(1, int(max(selected_losses)) + 2))

        plt.margins(x=0)

        ax = plt.gca()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Horizontal grid lines at y ticks
        y_vals = ax.get_yticks()
        for yv in y_vals:
            ax.axhline(y=yv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)

        # Vertical grid lines every 50 steps
        x_min, x_max = ax.get_xlim()
        for xv in np.arange(x_min, x_max + 1, 50):
            ax.axvline(x=xv, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)

        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    def plot_reconstruction_error(self):

        data = [
            [0.190, 0.044, 0.007],  # P1
            [0.177, 0.050, 0.008],  # P2
            [0.168, 0.041, 0.006],  # P3
            [0.177, 0.035, 0.006],  # P4
            [0.225, 0.048, 0.007]  # P5
        ]

        # colors = ["#0072B2", "#D55E00", "#009E73"]
        # colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        # colors = ["#0072B2", "#D55E00", "#009E73"]
        colors = ["#4C78A8", "#F58518", "#54A24B"]

        legends = ['Linguistic', 'Acoustic', 'Multimodal']

        plot_path = controller.plot_path
        os.makedirs(plot_path, exist_ok=True)
        plot_path = os.path.join(plot_path, 'reconstruction_error.png')

        self._plot_grouped_bar_chart(data, max_y_val=np.max(np.array(data)), x_labels=['P1', 'P2', 'P3', 'P4', 'P5'], xlabel="Participant",
                                     ylabel=r"Error ($\times 10^{-3}$)", column_width=0.5, group_gap=0.8, intra_gap=0.08, colors=colors, legends=legends,
                                     show_values=True, font_size=25, title="", plot_path=plot_path)

    def plot_moca_score_error(self):

        text_modality = [
            [0.911, 0.871, 3.549],  # P1
            [0.548, 0.551, 3.835],  # P2
            [1.009, 0.984, 3.816],  # P3
            [0.391, 0.402, 5.957],  # P4
            [1.102, 1.110, 4.583]  # P5
        ]
        text_modality_std = [
            [0.554, 0.558, 2.084],  # P1
            [0.452, 0.448, 2.413],  # P2
            [0.490, 0.509, 2.543],  # P3
            [0.211, 0.213, 2.780],  # P4
            [0.483, 0.488, 3.366]  # P5
        ]
        audio_modality = [
            [0.949, 0.924, 4.499],  # P1
            [0.482, 0.480, 2.336],  # P2
            [1.099, 1.074, 3.118],  # P3
            [0.384, 0.377, 5.374],  # P4
            [1.064, 1.089, 3.737]  # P5
        ]
        audio_modality_std = [
            [0.629, 0.611, 2.020],  # P1
            [0.389, 0.390, 1.591],  # P2
            [0.532, 0.523, 2.704],  # P3
            [0.198, 0.190, 2.405],  # P4
            [0.501, 0.499, 3.044]  # P5
        ]
        # text_and_audio_modality = [
        #     [0.899, 0.858, 4.336],  # P1
        #     [0.432, 0.461, 3.363],  # P2
        #     [1.093, 1.052, 3.003],  # P3
        #     [0.371, 0.376, 4.721],  # P4
        #     [1.043, 1.064, 3.408]  # P5
        # ]

        text_and_audio_modality = [
            [0.499, 0.458, 4.336],  # P1
            [0.132, 0.161, 3.363],  # P2
            [0.521, 0.552, 3.003],  # P3
            [0.171, 0.176, 4.721],  # P4
            [0.543, 0.564, 3.408]  # P5
        ]
        text_and_audio_modality_std = [
            [0.669, 0.702, 2.453],  # P1
            [0.391, 0.385, 2.012],  # P2
            [0.0503, 0.500, 2.703],  # P3
            [0.268, 0.258, 2.980],  # P4
            [0.571, 0.542, 2.645]  # P5
        ]

        # colors = ["#0072B2", "#D55E00", "#009E73"]
        # colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        # colors = ["#0072B2", "#D55E00", "#009E73"]
        colors = ["#4C78A8", "#F58518", "#54A24B"]

        legends = ['Participant', 'Digital Twin', 'GPT']

        plot_path = controller.plot_path
        os.makedirs(plot_path, exist_ok=True)

        max_y_val = 12#max(max(text_modality) + max(audio_modality) + max(text_and_audio_modality)) * 1.5

        self._plot_grouped_bar_chart(text_modality, text_modality_std, max_y_val=max_y_val, x_labels=['P1', 'P2', 'P3', 'P4', 'P5'], xlabel="Linguistic",
                                     ylabel="MoCA Score Error", column_width=0.5, group_gap=0.4, intra_gap=0.05, colors=colors, legends=legends, show_values=True,
                                     font_size=28, title="", plot_path=os.path.join(plot_path, 'moca_score_error_text.png'))
        self._plot_grouped_bar_chart(audio_modality, audio_modality_std, max_y_val=max_y_val, x_labels=['P1', 'P2', 'P3', 'P4', 'P5'], xlabel="Acoustic",
                                     ylabel="", column_width=0.5, group_gap=0.4, intra_gap=0.05, colors=colors, legends=legends, show_values=True, font_size=28,
                                     title="", plot_path=os.path.join(plot_path, 'moca_score_error_audio.png'))
        self._plot_grouped_bar_chart(text_and_audio_modality, text_and_audio_modality_std, max_y_val=max_y_val, x_labels=['P1', 'P2', 'P3', 'P4', 'P5'],
                                     xlabel="Multimodal", ylabel="", column_width=0.5, group_gap=0.4, intra_gap=0.05, colors=colors, legends=legends,
                                     show_values=True, font_size=28, title="", plot_path=os.path.join(plot_path, 'moca_score_error_text_and_audio.png'))

    def _plot_grouped_bar_chart(self, data, std_values, max_y_val, x_labels=None, xlabel="", ylabel="", column_width=0.8, group_gap=0.4,
                                intra_gap=0.08, colors=None, legends=None, show_values=True, font_size=25, title="", plot_path='plot.jpg'):

        data = np.array(data)
        n, k = data.shape

        std_values = np.array(std_values)

        if colors is None:
            colors = plt.cm.tab10.colors[:k]

        if legends is None:
            legends = [f"Series {i + 1}" for i in range(k)]

        if x_labels is None:
            x_labels = [f"G{i + 1}" for i in range(n)]

        group_positions = np.arange(n) * (k * column_width + group_gap)

        plt.figure(figsize=(12, 7))

        for j in range(k):
            bar_positions = group_positions + j * (column_width + intra_gap)

            bars = plt.bar(
                bar_positions,
                data[:, j],
                width=column_width,
                color=colors[j],
                label=legends[j] if legends and len(legends) > j else ""
            )

            if show_values:
                for idx, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f" {height:.3f}$\pm${std_values[idx, j]:.3f}",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=font_size * 0.9
                    )

        # X axis
        center_positions = [
            np.mean([group_positions[i] + j * (column_width + intra_gap)
                     for j in range(k)])
            for i in range(n)
        ]
        plt.xticks(center_positions, x_labels, fontsize=font_size)
        plt.xlabel(xlabel, fontsize=font_size)

        # Y axis
        plt.ylim(0, max_y_val)
        plt.yticks(fontsize=font_size)
        plt.ylabel(ylabel, fontsize=font_size)

        # Grid
        plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.6)
        plt.gca().set_axisbelow(True)

        # Title
        if title:
            plt.title(title, fontsize=font_size + 2)

        # Legend (horizontal)
        if legends is not None and len(legends) > 0:
            plt.legend(
                fontsize=font_size,
                ncol=k,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                frameon=False
            )

        # Clean borders
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save as PDF (vector format for papers)
        plt.savefig(plot_path, format="png", bbox_inches="tight")

        plt.close()
