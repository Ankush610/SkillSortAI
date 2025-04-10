from __future__ import annotations
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import process_resumes,save_resumes_to_excel
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time


# Placeholder for your functions
# from your_module import process_resumes, save_resumes_to_excel

global_df = pd.DataFrame()

def extract_and_rank(requirements, resume_dir, top_n):

    """Helper Function to take User Input from Gradio Interface and return Output"""
    
    global global_df
    output = process_resumes(requirements, resume_dir)
    df = save_resumes_to_excel(output)

    # Convert scores to numeric
    for col in ['Overall Score', 'Skills Score', 'Experience Score', 'Behavior Index']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_sorted = df.sort_values(by='Overall Score', ascending=False).head(int(top_n))
    global_df = df_sorted.reset_index(drop=True)

    return df_sorted, *generate_all_plots(df_sorted)

def generate_all_plots(df):

    """Helper Function to Generate Plots for the Gradio Interface"""
    
    def plot_scores(score_column, title):
        # Dynamic height: 0.5 inch per row, minimum 4 inches
        height = max(4, 0.5 * len(df))
        fig, ax = plt.subplots(figsize=(10, height))
        colors = sns.color_palette("husl", len(df))
        ax.barh(df['Name'], df[score_column], color=colors)
        ax.set_xlabel(score_column)
        ax.set_title(title)
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    return (
        plot_scores('Overall Score', 'Overall Score'),
        plot_scores('Skills Score', 'Skills Score'),
        plot_scores('Experience Score', 'Experience Score'),
        plot_scores('Behavior Index', 'Behavior Index')
    )

def update_plots_on_selection(selected_rows):
    global global_df
    if not selected_rows:
        return generate_all_plots(global_df)
    selected_df = global_df.iloc[selected_rows]
    return generate_all_plots(selected_df)

class Seafoam(Base):

    """Theme Class for Visual Changes to Gradio Interface"""
    
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

with gr.Blocks(theme=Seafoam()) as demo:
    
    gr.Markdown("<center><h1>🤖--|| SkillSort AI ||--🤖<h1><center>")  

    with gr.Row():
        textbox = gr.Textbox(placeholder="Write job requirements", label="Job Requirements", lines=3)
        textbox_2 = gr.Textbox(label="Resume Directory Path", placeholder="Enter path to resumes")

    with gr.Row():
        slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Select top N records")

    with gr.Row():
        button = gr.Button(value="Extract and Rank Resumes", variant='primary')
        clear_button = gr.Button(value="Clear Inputs", variant='secondary')

    dataframe = gr.Dataframe(label="Resume Results", interactive=False, height=300)

    with gr.Row():
        plot1 = gr.Plot(label="Overall Score")
        plot2 = gr.Plot(label="Skills Score")

    with gr.Row():
        plot3 = gr.Plot(label="Experience Score")
        plot4 = gr.Plot(label="Behavior Index")

    # Connect main button
    button.click(
        fn=extract_and_rank,
        inputs=[textbox, textbox_2, slider],
        outputs=[dataframe, plot1, plot2, plot3, plot4]
    )

    # Add functionality to clear inputs and outputs
    def clear_inputs():
        return "", "", None, None, None, None, pd.DataFrame()

    clear_button.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[textbox, textbox_2]
    )

    # Handle row selection for updated plots
    @dataframe.select()
    def on_select(evt: gr.SelectData):
        if evt.index is None:
            return generate_all_plots(global_df)
        return update_plots_on_selection([evt.index])

demo.launch(share=True)
