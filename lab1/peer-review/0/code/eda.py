#package imports

import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import math 
import textwrap
import numpy as np



def plot_value_counts(df, variable_metadata, columns, df2 = None, ncols=3, labels = None, figsize_per_plot=(5, 4), color="teal", fontsize = 10):
    """
    Plot value counts (bar plot) for multiple columns in a dataframe,
    each as a separate subplot within a single figure.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the columns to plot.
    variable_metadata : pandas.DataFrame
        Dataframe containing the metadata (explanations) of each variable
    columns : list of str
        List of column names to plot.
    ncols : int, optional
        Number of subplot columns in the grid. Default is 3.
    figsize_per_plot : tuple, optional
        Size (width, height) allotted per subplot. Default is (5, 4).
    color : str, optional
        Color for the bars. Default is "blue".
    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes objects
    """
    n = len(columns)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )

    # flatten axes array 
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(columns):

        #Assign label based on integer value recorded
        values_key = variable_metadata['values_key'][col]
        values_key = values_key.strip().split('\n')
        values_key = values_key[1:]
            
        label_dict = {}

        for line in values_key:
            parts = line.split(maxsplit=1)  # split into code and label 
            code, label = int(parts[0]), parts[1]
            label_dict[code] = label.strip()

        counts = df[col].value_counts(dropna = False)

        if df2 is not None:
            # get counts for the second dataframe and align both on the same set of values
            counts2 = df2[col].value_counts(dropna = False)
            all_values = counts.index.union(counts2.index, sort=False)

            counts = counts.reindex(all_values, fill_value=0)
            counts2 = counts2.reindex(all_values, fill_value=0)

            x = np.arange(len(counts))
            width = 0.4

            bars1 = axes[i].bar(x - width/2, counts.values, width=width, label=labels[0], color = 'darkseagreen', zorder=3)
            bars2 = axes[i].bar(x + width/2, counts2.values, width=width, label=labels[1], color = 'cornflowerblue', zorder=3)
            axes[i].legend(fontsize=fontsize)
            axes[i].grid(axis='y', color='lightgray', linewidth=0.3)
            axes[i].tick_params(direction='in')

        else:


            axes[i].bar(range(len(counts)), counts.values, color='darkseagreen', zorder=3)
        axes[i].set_title(textwrap.fill(str(variable_metadata['description'][col]), width=45), fontsize = fontsize)
        # axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count", fontsize = fontsize)

        # Replace tick labels with text from label_dict 
        tick_labels = [f"Not\n Recorded" if pd.isna(val) else textwrap.fill(str(label_dict.get(val, val)), width=13)
            for val in counts.index]
            # textwrap.fill(str(label_dict.get(val, val)), width=12) for val in counts.index]
        axes[i].set_xticks(range(len(counts)))
        axes[i].set_xticklabels(tick_labels, ha='center', fontsize = fontsize)
        axes[i].grid(axis='y', color='lightgray', linewidth=0.3)
        axes[i].tick_params(direction='in', labelsize = fontsize)
        

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_missing_vals(df, df2=None, labels=None, xlim=18000, figsize=(6, 10), plot_title=None, max_vars=None, variable_metadata=None):
    """
    Plot the null values as a horizontal bar plot, ordered from smallest to
    largest missing count.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data
    df2 : pandas.DataFrame, optional
        Optional second dataset to overlay the first one
    labels : list of str, optional
        Legend labels for df and df2, required if df2 is provided
    xlim : int or float, optional
        Upper limit for the x-axis (count). Default is 18000.
    figsize : tuple, optional
        Figure size (width, height). Default is (6, 10).
    plot_title : string, optional
        Title of Plot
    max_vars : int, optional
        Maximum number of variables to display, showing those with the
        highest missing counts. If None, all variables with missing
        values are shown.
    variable_metadata : pandas.DataFrame, optional
        DataFrame indexed by variable name, with a 'variable_label' column
        used to generate human-readable y-axis labels. If None, raw
        column names are used instead.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    # Sum missing values for each variable
    missing_vals = df.isnull().sum()

    # Build a label lookup, falling back to the raw variable name if missing
    if variable_metadata is not None:
        label_map = variable_metadata['variable_label'].to_dict()
    else:
        label_map = {}

    if df2 is not None:
        missing_vals2 = df2.isnull().sum()

        # Union of variables with missing values in either dataframe
        variables = missing_vals[missing_vals > 0].index.union(missing_vals2[missing_vals2 > 0].index)

        vals1 = missing_vals.reindex(variables, fill_value=0)
        vals2 = missing_vals2.reindex(variables, fill_value=0)

        # Combine into a long-form DataFrame for seaborn
        plot_df = pd.DataFrame({
            'Variable': list(variables) * 2,
            'Count': list(vals1) + list(vals2),
            'Dataset': [labels[0]] * len(variables) + [labels[1]] * len(variables)
        })

        # Rank variables by their max count across both datasets
        total_by_var = plot_df.groupby('Variable')['Count'].max().sort_values(ascending=True)

        # Limit to top N variables (by highest missing count) if requested
        if max_vars is not None:
            total_by_var = total_by_var.tail(max_vars)
            plot_df = plot_df[plot_df['Variable'].isin(total_by_var.index)]

        order = total_by_var.index.tolist()
        display_labels = [label_map.get(v, v) for v in order]

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(axis='x', color='lightgray', linewidth=0.3, zorder=0)
        sns.barplot(
            data=plot_df,
            y='Variable',
            x='Count',
            hue='Dataset',
            order=order,
            palette=['darkseagreen', 'cornflowerblue'],
            edgecolor='white',
            linewidth=0.5,
            ax=ax,
            zorder=3
        )

        for container in ax.containers:
            ax.bar_label(container, padding=2, fmt='%.0f')

        ax.set_yticklabels(display_labels)
        ax.legend()

    else:
        # Filter to only the variables that have missing values
        missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=True)

        # Limit to top N variables (by highest missing count) if requested
        if max_vars is not None:
            missing_vals = missing_vals.tail(max_vars)

        plot_df = missing_vals.reset_index()
        plot_df.columns = ['Variable', 'Count']
        display_labels = [label_map.get(v, v) for v in plot_df['Variable']]

        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(axis='x', color='lightgray', linewidth=0.3, zorder=0)
        sns.barplot(
            data=plot_df,
            y='Variable',
            x='Count',
            order=plot_df['Variable'],
            color='darkseagreen',
            edgecolor='white',
            linewidth=0.5,
            ax=ax,
            zorder=3
        )

        for container in ax.containers:
            ax.bar_label(container, padding=2, fmt='%.0f')

        ax.set_yticklabels(display_labels)

    ax.set_xlim(0, xlim)
    ax.set_xlabel('Count', fontsize=11, labelpad=5)
    ax.set_ylabel('Variable', fontsize=11, labelpad=5)
    ax.set_title(plot_title, loc='center', fontsize=12, fontweight='bold', pad=10)

    ax.tick_params(direction='in')
    ax.tick_params(left=False)

    return fig, ax

def get_pct_summary_given_ciTBI(data, predictors, target='PosIntFinal', severity_vars=None):
    """Return % predictor=observed among patients where target=1 (i.e., ciTBI observed).
    Uses cond_sev_obs (3) as the observed value for variables in severity_vars,
    and cond_yes_no (1) for all other variables."""

    cond_sev_obs = 3  # High severity for injury, headache
    cond_yes_no = 1   # A condition was observed

    severity_vars = severity_vars or []
    summary_pct = {}
    for variable in predictors:
        cross_tab = pd.crosstab(data[variable], data[target], normalize='columns') * 100
        obs_val = cond_sev_obs if variable in severity_vars else cond_yes_no
        if obs_val in cross_tab.index and 1 in cross_tab.columns:
            summary_pct[variable] = cross_tab.loc[obs_val, 1]
        else:
            summary_pct[variable] = np.nan
    return summary_pct

def get_pct_summary_given_predictor(data, predictors, target='PosIntFinal', severity_vars=None):
    """Return % target=1 among patients where predictor=observed.
    Uses cond_sev_obs (3) as the observed value for variables in severity_vars,
    and cond_yes_no (1) for all other variables."""

    cond_sev_obs = 3  # High severity for injury, headache
    cond_yes_no = 1   # A condition was observed

    severity_vars = severity_vars or []
    summary_pct = {}
    for variable in predictors:
        cross_tab = pd.crosstab(data[variable], data[target], normalize='index') * 100
        obs_val = cond_sev_obs if variable in severity_vars else cond_yes_no
        if obs_val in cross_tab.index and 1 in cross_tab.columns:
            summary_pct[variable] = cross_tab.loc[obs_val, 1]
        else:
            summary_pct[variable] = np.nan
    return summary_pct

def variable_tbi_heatmap(df, variables_to_plot, variable_metadata, plot_title, split_by_age=False, cmap='OrRd', vmin=0, vmax=100, severity_vars=None, fontsize = 12):
    """
    Plot a heatmap showing the percent of patients with a positive ciTBI 
    finding for each predictor variable split into age bins.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    variables_to_plot : list of str
        Column names in `df` to include as predictors in the heatmap.
    variable_metadata : pd.DataFrame
        variable_metadata, used to generate heatmap
    cmap : str, default 'flare'
    vmin : int, default 0
        min value for heatmap scale
    vmax : int, default 100
        max value for heatmap scale
    severity_vars : list of str, optional
        variable names with a severity range rather than a yes/no format
        (cond_sev_obs) instead of the standard "observed" threshold
        (cond_yes_no) in get_pct_summary

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if severity_vars is None:
        severity_vars = []

    if split_by_age:
        # --- Define age bins ---
        bin_edges = [-np.inf, 2, 4, 6, 10, np.inf]
        bin_labels = ['Below 2', '2 to 4', '4 to 6', '6 to 10', '10+']

        df = df.copy()
        df['_age_bin'] = pd.cut(df['AgeinYears'], bins=bin_edges, labels=bin_labels, right=False)

        group_pcts = {}
        for label in bin_labels:
            group_df = df[df['_age_bin'] == label]
            group_pcts[label] = get_pct_summary_given_ciTBI(group_df, variables_to_plot, severity_vars=severity_vars)

        summary_df = pd.DataFrame(group_pcts)
    else:
        # No age split — single column using the full dataset
        pct_all = get_pct_summary_given_ciTBI(df, variables_to_plot, severity_vars=severity_vars)
        summary_df = pd.DataFrame({'All ages': pct_all})

    # Sort by the max across columns so most "important" predictors are on top
    summary_df = summary_df.loc[summary_df.max(axis=1).sort_values(ascending=False).index]

    # --- Map variable names to descriptions from variable_metadata ---
    label_map = variable_metadata['variable_label'].to_dict()

    # Update variables with "severity" to specifically be "severe"
    if severity_vars is not None:
        custom_labels = {
            'HASeverity': 'Severe headache?',
            'High_impact_InjSev': 'Severe Injury?'
        }
        label_map.update(custom_labels)
    y_labels = [label_map.get(var, var) for var in summary_df.index]

    # --- Build custom annotation labels (numeric or "Not recorded") ---
    annot_labels = summary_df.map(lambda x: 'Not\nrecorded' if pd.isna(x) else f'{x:.1f}%')

    # Fill NaNs with 0 just for coloring purposes — annotation text still shows "Not recorded"
    summary_df_filled = summary_df.fillna(0)

    # --- Heatmap (seaborn) ---
    fig, ax = plt.subplots(figsize=(8,8))

    sns.heatmap(
        summary_df_filled,
        annot=annot_labels,
        fmt='',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        yticklabels=y_labels,
        cbar_kws={'label': '% patients with ciTBI'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.5)

    ax.set_xlabel('Age', labelpad = 3, fontsize = fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
 

    plt.title(plot_title, fontsize=fontsize, fontweight='bold')
    plt.tight_layout()
    return fig, ax


def plot_box_whiskers(df, variable_predictors, variable_to_plot, variable_metadata=None, figsize=None, fontsize = 12):
    """
    Plot a grid of box-and-whisker plots showing the age distribution,
    split by whether each condition (predictor) was observed.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    variable_predictors : list of str
        Column names to plot age distributions for.
    variable_metadata : pd.DataFrame, optional
        DataFrame indexed by variable name with 'variable_label' (for subplot
        titles) and 'values_key' (for x-axis tick labels) columns.
    variable_to_plot : str
        Name of the numeric column.
    figsize : tuple, optional
        Figure size. Defaults to scaling with the number of variables.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    """
    n_vars = len(variable_predictors)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    label_map = variable_metadata['variable_label'].to_dict() if variable_metadata is not None else {}

    for i, variable in enumerate(variable_predictors):
        sns.boxplot(data=df, x=variable, y=variable_to_plot, ax=axes[i], color='cornflowerblue')

        title = label_map.get(variable, variable)
        axes[i].set_title(title, fontsize=fontsize)
        axes[i].set_xlabel('')
        axes[i].set_ylabel(variable_to_plot, fontsize = fontsize)

        # Replace x-axis tick labels using the values_key mapping
        if variable_metadata is not None and 'values_key' in variable_metadata.columns:
            # Assign label based on integer value recorded
            values_key = variable_metadata['values_key'][variable]
            values_key = values_key.strip().split('\n')
            values_key = values_key[1:]

            label_dict = {}
            for line in values_key:
                parts = line.split(maxsplit=1)  # split into code and label
                code, label = int(parts[0]), parts[1]
                label_dict[code] = label.strip()

            current_ticks = axes[i].get_xticks()
            #convert the labels from integer values to categories 
            current_labels = [int(float(t.get_text())) for t in axes[i].get_xticklabels()]
            #add line breaks for long category names
            new_labels = [label_dict.get(code, code) for code in current_labels]
            new_labels = pd.Series(new_labels).str.replace(r'\s*\([^)]*\)', '', regex=True).tolist()
            # Add line breaks for long category names
            new_labels = [textwrap.fill(str(label), width=10) for label in new_labels]
            axes[i].set_xticks(current_ticks)
            axes[i].set_xticklabels(new_labels, ha='center', fontsize = fontsize)
            axes[i].tick_params(axis='both', labelsize=fontsize)


    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig, axes


def plot_findings(df, variable_metadata, patient_filter_col, plot_title, top_n=8, ax=None, fontsize = 12):
    patient_filter = df[df[patient_filter_col] == 1]

    if patient_filter_col == 'CTDone':
        finding_cols = [f'Finding{i}' for r in [range(1, 15), range(20, 24)] for i in r]
        finding_counts = patient_filter[finding_cols].sum().sort_values(ascending=False)

        if top_n is not None:
            finding_counts = finding_counts[0:top_n]

        finding_df = finding_counts.reset_index()
        finding_df.columns = ['Finding', 'Count']

        label_map = (
            variable_metadata['description']
            .str.replace('Traumatic finding: ', '', regex=False)
            .str.replace(' ', '\n', regex=False)
            .str.replace('/', '\n/', regex=False)
            .to_dict()
        )

    elif patient_filter_col == 'CTForm1':
        finding_cols = df.columns[df.columns.str.contains('Ind')].tolist()
        finding_counts = patient_filter[finding_cols].sum().sort_values(ascending=False)

        if top_n is not None:
            finding_counts = finding_counts[0:top_n]

        finding_df = finding_counts.reset_index()
        finding_df.columns = ['Finding', 'Count']

        label_map = (
            variable_metadata['description']
            .str.replace('If a CT is being ordered or obtained check the most important indications in influencing the decision to obtain a head CT: ', '', regex=False)
            .str.replace(' ', '\n', regex=False)
            .str.replace('/', '\n/', regex=False)
            .to_dict()
        )

    else:
        raise ValueError(f"Unsupported patient_filter_col: {patient_filter_col}")

    finding_df['Finding_Label'] = finding_df['Finding'].map(lambda x: label_map.get(x, x))

    # Use provided ax if given, otherwise create a new standalone figure
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(finding_df)), 6))
    else:
        fig = ax.figure

    ax.grid(axis='y', color='lightgray', linewidth=0.5, zorder=0)

    sns.barplot(data=finding_df, x='Finding_Label', y='Count', color='cornflowerblue', ax=ax, order=finding_df['Finding_Label'], zorder=3)

    ax.bar_label(ax.containers[0], padding=2, fontsize = fontsize)
    ax.set_xlabel('Findings', fontsize = fontsize)
    ax.set_ylabel('Number of Patients', fontsize = fontsize)
    ax.set_title(plot_title, fontsize=fontsize, fontweight='bold')
    ax.set_ylim(0, finding_df['Count'].max() * 1.15)

    ax.tick_params(axis='x', rotation=0, direction='in')
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='both', labelsize=fontsize)

    if standalone:
        plt.tight_layout()
        plt.show()

    return fig, ax


def plot_ct_yield_by_indication(df, variable_metadata, top_n=8, xmax=100, fontsize = 12):
    ind_cols = df.columns[df.columns.str.contains('Ind')].tolist()
    finding_cols = [f'Finding{i}' for r in [range(1, 15), range(20, 24)] for i in r]

    yield_pct = {}
    for ind in ind_cols:
        subset = df[df[ind] == 1]
        if len(subset) > 0:
            had_any_finding = (subset[finding_cols].sum(axis=1) > 0).mean() * 100
            yield_pct[ind] = had_any_finding

    yield_df = pd.Series(yield_pct).reset_index()
    yield_df.columns = ['Variable', 'Percent']

    label_map = variable_metadata['variable_label'].to_dict()
    yield_df['Label'] = yield_df['Variable'].map(lambda x: label_map.get(x, x))

    yield_df = yield_df.sort_values('Percent', ascending=True)
    if top_n:
        yield_df = yield_df.tail(top_n)

    order = yield_df['Label'].tolist()
    kept_vars = yield_df['Variable'].tolist()

    # Build a long-form age dataframe: one row per patient per indication (where indication == 1)
    age_rows = []
    for var in kept_vars:
        subset = df[df[var] == 1]
        label = label_map.get(var, var)
        for age_val in subset['AgeinYears'].dropna():
            age_rows.append({'Label': label, 'Age': age_val})
    age_df = pd.DataFrame(age_rows)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.4 * len(yield_df) + 1), sharey=True)

    # Left: % with any positive finding
    axes[0].grid(axis='x', color='lightgray', linewidth=0.5, zorder=0)
    sns.barplot(data=yield_df, y='Label', x='Percent', order=order, color='mediumseagreen', ax=axes[0], zorder=3)
    axes[0].bar_label(axes[0].containers[0], fmt='%.1f%%', padding=3)
    axes[0].set_xlabel('% with ciTBI finding(s)', fontsize = fontsize)
    axes[0].set_ylabel('')
    axes[0].set_xlim(0, xmax)
    axes[0].set_title('ciTBI Percent Based on Reason CT Scan Ordered', fontsize=fontsize, fontweight='bold')
    axes[0].tick_params(direction='in', labelsize = fontsize)

    # Right: Age distribution per indication
    axes[1].grid(axis='x', color='lightgray', linewidth=0.5, zorder=0)
    sns.boxplot(data=age_df, y='Label', x='Age', order=order, color='cornflowerblue', ax=axes[1], zorder=3)
    axes[1].set_xlabel('Age (years)', fontsize = fontsize)
    axes[1].set_ylabel('', fontsize = fontsize)
    axes[1].set_title('Age Distribution by Reason CT Scan Ordered', fontsize=fontsize, fontweight='bold')
    axes[1].tick_params(direction='in', labelsize = fontsize)
    axes[1].tick_params(labelleft=True)

    plt.tight_layout()
    return fig, axes