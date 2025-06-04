import os
from datetime import datetime
import math

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

pyo.init_notebook_mode()


def plot_data(
    data: pd.DataFrame,
    title: str = "Data Plot",
    yaxis_title: str = "Value",
    mode="lines",
    annotation_columns: list[str] | None = None,
    hidden_columns: list[str] | None = None,
) -> None:
    """Plot pandas data using Plotly with optional annotations."""
    fig = go.Figure()

    for column in data.columns:
        if annotation_columns is None or column not in annotation_columns:
            visible = (
                "legendonly"
                if hidden_columns and column in hidden_columns
                else True
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode=mode,
                    name=column,
                    visible=visible,
                )
            )

    if annotation_columns is not None:
        for annotation_column in annotation_columns:
            if annotation_column in data.columns:
                annotation_dates = []
                annotation_values = []
                annotation_texts = []

                for date, value in data[data[annotation_column]].iterrows():
                    annotation_dates.append(date)
                    annotation_values.append(value[data.columns[0]])
                    annotation_texts.append(annotation_column)

                fig.add_trace(
                    go.Scatter(
                        x=annotation_dates,
                        y=annotation_values,
                        text=annotation_texts,
                        mode="markers+text",
                        marker=dict(size=10, standoff=4),
                        textposition="top center",
                        name=annotation_column,
                        showlegend=True,
                        visible=(
                            "legendonly"
                            if hidden_columns
                            and annotation_column in hidden_columns
                            else True
                        ),
                    )
                )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        legend_title="Columns",
    )

    fig.show()


def plot_data_dual_y_axis(
    data: pd.DataFrame,
    column1: str,
    column2: str,
    title: str = "Data Plot",
    mode="lines",
) -> None:
    """Plot two columns of pandas data using Plotly with separate y-axes."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column1],
            mode=mode,
            name=column1,
            line=dict(color="blue"),
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column2],
            mode=mode,
            name=column2,
            line=dict(color="red"),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis=dict(
            title=column1,
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title=column2,
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right",
        ),
        legend_title="Columns",
    )

    fig.show()


def plot_3d_surface(
    data: pd.DataFrame, x_col_name: str, y_col_name: str, z_col_name: str
) -> None:
    """
    Plot a 3D surface plot using specified columns from the DataFrame.

    Parameters:
    data (pd.DataFrame): The input data containing the columns to plot.
    x_col_name (str): The name of the column to use for the x-axis.
    y_col_name (str): The name of the column to use for the y-axis.
    z_col_name (str): The name of the column to use for the z-axis.
    """
    pivot_table = data.pivot(
        index=y_col_name, columns=x_col_name, values=z_col_name
    )

    x_values = pivot_table.columns.values
    y_values = pivot_table.index.values
    z_values = pivot_table.values

    fig = go.Figure(data=[go.Surface(x=x_values, y=y_values, z=z_values)])

    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.5),
        center=dict(x=0, y=0, z=-0.5),
        up=dict(x=0, y=0, z=1),
    )

    fig.update_layout(
        title="3D Surface Plot",
        scene=dict(
            xaxis_title=x_col_name,
            yaxis_title=y_col_name,
            zaxis_title=z_col_name,
            camera=camera,
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig.show()


def split_data(
    data: pd.DataFrame, offset: int, ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = offset + math.ceil((data.shape[0] - offset) * ratio)
    training_df = data.iloc[offset:split_index]
    testing_df = data.iloc[split_index:]
    return training_df, testing_df


def get_absolute_path(method: str = "module", relative_path: str = "") -> str:
    """
    Get the absolute path relative to the directory of the calling module or
    current working directory.

    Parameters:
    method (str): Method to resolve the path, "module" or "cwd".
    relative_path (str): The relative path from the directory.

    Returns:
    str: The absolute path.
    """
    if method == "module":
        calling_module_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.join(calling_module_dir, relative_path)
        return os.path.abspath(absolute_path)
    elif method == "cwd":
        absolute_path = os.path.join(os.getcwd(), relative_path)
        return os.path.abspath(absolute_path)
    else:
        raise ValueError("Invalid method. Use 'module' or 'cwd'.")


def read_data(
    file_name: str, file_ext: str = "csv", dir_path: str = ""
) -> pd.DataFrame:
    """
    Use pandas to read a specific file.

    Parameters:
    file_name (str): The name of the file without extension.
    file_ext (str): The file extension, default is 'csv'.
    dir_path (str): The directory path where the file is located.

    Returns:
    pd.DataFrame: The data read by pandas.

    Raises:
    ValueError: If the file extension is unsupported.
    """
    file_path = os.path.join(dir_path, f"{file_name}.{file_ext}")

    if file_ext == "csv":
        return pd.read_csv(file_path)
    elif file_ext == "parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")


def convert_to_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%d/%m/%Y")


def convert_to_array(
    input: npt.ArrayLike, dtype: str = "float64"
) -> np.ndarray:
    """
    Convert input to a numpy array of the specified dtype.

    Parameters:
    input (npt.ArrayLike): The input data to convert.
    dtype (str): The desired data type of the output array. Must be 'float64'
    or 'int64'. Default is 'float64'.

    Returns:
    np.ndarray: The converted numpy array.

    Raises:
    ValueError: If the dtype is not 'float64' or 'int64'.
    """
    valid_dtypes = ["float64", "int64"]
    if dtype not in valid_dtypes:
        raise ValueError(f"dtype must be one of {valid_dtypes}")
    desired_dtype = np.float64 if dtype == "float64" else np.int64
    if not (isinstance(input, np.ndarray) and input.dtype == desired_dtype):
        output = np.array(object=input, dtype=desired_dtype)
    else:
        output = input
    return output
