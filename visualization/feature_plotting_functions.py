# %% [markdown]
# PhD team 
# Presenting Code for PhD Thesis

# %% info [markdown]
# Here, we are presenting a completed routine, a thorough connected blocks, to perform the very idea of my Ph.D. thesis, conducting a comprehensively automatic and longitudinal analyses of a given MS patient.
# All of the rights of this routine are reserved for the developer(s).
# 
# 
# Mahdi Bashiri Bawil
# Developer

# %% [markdown]
# 

# %% [markdown]
# # Attempt : Comprehensive Analyses 

# %% [markdown]
# %% [markdown]
# ## Phase 0: Dependencies & Functions

# %% [markdown]
# ### Packages

# %% Packages
import os
import cv2
import json
import skimage
import numpy as np
import plotly.io as pio
from datetime import datetime
import plotly.graph_objs as go
import matplotlib.pyplot as plt


# %%                        Default Configurations for Visualizations
# color_codes should be defined globally or passed as parameter
font_family = "Arial"
title_font_size = 18
axis_font_size = 18
legend_font_size = 18
annotation_font_size = 20

# Use the same color scheme and font settings as the count function
"""    
color_codes = {
    'peri': '#2E86AB',      # Professional blue
    'para': '#A23B72',      # Professional magenta
    'juxt': '#F18F01',      # Professional orange
    'total': '#2F4858'      # Dark blue-gray for annotations
}"""


# %% Plotting Functions of Five Features

#
def num_show_subj(save_path, wmh_num, wmh_code, id, tp=0):
    """
    Enhanced function for publication-ready WMH lesion visualization with JSON data saving
    """
    
    """    # Define publication-ready color scheme
    color_codes = {
        'peri': '#2E86AB',      # Professional blue
        'para': '#A23B72',      # Professional magenta
        'juxt': '#F18F01',      # Professional orange
        'total': '#2F4858'      # Dark blue-gray for annotations
    }"""

    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        total_value = 0
        print(f"There is no seen plaque.")
    else:
        total_value = np.sum(wmh_num)
        print(f"Total Number of Plaques: {total_value}")

    # ==================== SCATTER PLOT (First visualization) ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        fig = go.Figure().update_layout(
            height=600,
            width=1100,
            xaxis_title='Slices',
            yaxis_title='Frequency of Plaques in Slices',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))

        # Save JSON data for slice-wise scatter plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_number_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_plaques": 0,
            },
            "data": {
                "slice_numbers": [],
                "plaques_per_slice": []
            }
        }

    else:
        x_values = np.arange(1, len(wmh_num) + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=wmh_num,
            mode='markers',
            marker=dict(
                size=8,
                color=color_codes['total'],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name='Plaque Count'
        ))
        
        fig.update_layout(
            height=600, width=1100,
            xaxis_title='Slice Number',
            yaxis_title='Number of Plaques per Slice',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            showlegend=False
        )
        
        fig.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Number of Plaques: {total_value}",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))
        
        # Save JSON data for slice-wise scatter plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_number_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_num),
                "total_plaques": int(total_value),
                    "mean_plaques_per_slice": float(np.mean(wmh_num)),
                    "max_plaques_per_slice": int(np.max(wmh_num)),
                    "std_plaques_per_slice": float(np.std(wmh_num))
            },
            "data": {
                "slice_numbers": x_values.tolist(),
                "plaques_per_slice": [int(count) for count in wmh_num]
            }
        }

    # Save scatter plot image
    pio.write_image(fig, os.path.join(save_path, f'Slices_Plaque_Number_tp{tp}.png'), 
                    width=1100, height=600, scale=2)
    
    # Save scatter plot data as JSON
    with open(os.path.join(save_path, f'slice_wise_number_scatter_data_tp{tp}.json'), 'w') as f:
        json.dump(scatter_slices_data, f, indent=2)

    # ==================== CALCULATE CATEGORY COUNTS ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        num_peri = []
        num_para = []
        num_juxt = []
    else:
        peri = np.where(np.array(wmh_code) == 1, 1, 0)
        para = np.where(np.array(wmh_code) == 2, 1, 0)
        juxt = np.where(np.array(wmh_code) == 3, 1, 0)
        
        num_peri = np.zeros_like(np.array(wmh_num))
        num_para = np.zeros_like(np.array(wmh_num))
        num_juxt = np.zeros_like(np.array(wmh_num))
        
        i = 0
        c = 0
        for k in wmh_num:
            if k == 0:
                c += 1
                continue
            num_peri[c] = np.sum(peri[i:i+k])
            num_para[c] = np.sum(para[i:i+k])
            num_juxt[c] = np.sum(juxt[i:i+k])
            c += 1
            i += k

    # ==================== GROUPED BAR PLOT FOR COUNTS ====================
    if len(num_peri) == 0 or len(num_para) == 0 or len(num_juxt) == 0:
        plot = go.Figure().update_layout(
            height=600, width=1200,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        plot.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
                "periventricular_total": 0,
                "paraventricular_total": 0,
                "juxtacortical_total": 0
            },
            "data": {
                "slice_numbers": [],
                "periventricular_numbers": [],
                "paraventricular_numbers": [],
                "juxtacortical_numbers": []
            }
        }

    else:
        x = list(range(1, len(wmh_num) + 1))  # Start from 1 for slice numbers
        
        plot = go.Figure()
        
        # Create grouped bars with offset positions
        bar_width = 0.25
        
        plot.add_trace(go.Bar(
            name='Periventricular',
            x=[xi - bar_width for xi in x], 
            y=list(num_peri),
            marker=dict(color=color_codes['peri']),
            width=bar_width,
            offsetgroup=1
        ))
        
        plot.add_trace(go.Bar(
            name='Paraventricular',
            x=x, 
            y=list(num_para),
            marker=dict(color=color_codes['para']),
            width=bar_width,
            offsetgroup=2
        ))
        
        plot.add_trace(go.Bar(
            name='Juxtacortical',
            x=[xi + bar_width for xi in x], 
            y=list(num_juxt),
            marker=dict(color=color_codes['juxt']),
            width=bar_width,
            offsetgroup=3
        ))
        
        plot.update_layout(
            barmode='group',
            height=600, width=1200,
            xaxis_title='Slice Number',
            yaxis_title='Number of Plaques per Slice',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5,
                tick0=1,
                dtick=1  # Ensure integer ticks for slice numbers
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            # Position legend inside plot area at upper left
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            ),
            bargap=0.2,  # Gap between groups of bars
            bargroupgap=0.1  # Gap between bars within a group
        )
        
        # Save JSON data for grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_num),
                "periventricular_total": float(np.sum(num_peri)),
                "paraventricular_total": float(np.sum(num_para)),
                "juxtacortical_total": float(np.sum(num_juxt)),
                "slices_with_peri": int(np.sum(num_peri > 0)),
                "slices_with_para": int(np.sum(num_para > 0)),
                "slices_with_juxt": int(np.sum(num_juxt > 0))
            },
            "data": {
                "slice_numbers": x,
                "periventricular_numbers": [int(count) for count in num_peri],
                "paraventricular_numbers": [int(count) for count in num_para],
                "juxtacortical_numbers": [int(count) for count in num_juxt],
                "bar_positions": {
                    "periventricular_x": [xi - bar_width for xi in x],
                    "paraventricular_x": x,
                    "juxtacortical_x": [xi + bar_width for xi in x]
                }
            }
        }

    # Save grouped bar plot image
    pio.write_image(plot, os.path.join(save_path, f'Category_Plaque_Number_tp{tp}.png'),
                    width=1200, height=600, scale=2)
    
    # Save grouped bar plot data as JSON
    with open(os.path.join(save_path, f'category_number_grouped_bar_data_tp{tp}.json'), 'w') as f:
        json.dump(grouped_bar_data, f, indent=2)

    # ==================== PIE CHART ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0 or (np.sum(num_peri) == 0 and np.sum(num_para) == 0 and np.sum(num_juxt) == 0):
        fig = go.Figure().update_layout(
            height=600, width=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No data to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save pie chart data as JSON (empty case)
        pie_chart_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "pie_chart_number_distribution",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_number": 0,
                "categories_present": 0
            },
            "data": {
                "categories": [],
                "values": [],
                "percentages": [],
                "colors": []
            }
        }
    else:
        # Calculate totals for each category
        peri_total = np.sum(num_peri) if len(num_peri) > 0 else 0
        para_total = np.sum(num_para) if len(num_para) > 0 else 0
        juxt_total = np.sum(num_juxt) if len(num_juxt) > 0 else 0
        
        # Only include categories with non-zero values
        groups = []
        values = []
        colors = []
        
        if peri_total > 0:
            groups.append('Periventricular')
            values.append(peri_total)
            colors.append(color_codes['peri'])
            
        if para_total > 0:
            groups.append('Paraventricular')
            values.append(para_total)
            colors.append(color_codes['para'])
            
        if juxt_total > 0:
            groups.append('Juxtacortical')
            values.append(juxt_total)
            colors.append(color_codes['juxt'])
        
        if len(groups) == 0:  # All values are zero
            fig = go.Figure().update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size)
            )
            fig.add_annotation(dict(
                font=dict(color='gray', size=18, family=font_family),
                x=0.5, y=0.5, showarrow=False,
                text="No data to display", textangle=0,
                xanchor='center', xref="paper", yref="paper"
            ))
            
            # Save pie chart data as JSON (all zeros case)
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_number_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": False
                },
                "summary_statistics": {
                    "total_number": 0,
                    "categories_present": 0
                },
                "data": {
                    "categories": [],
                    "values": [],
                    "percentages": [],
                    "colors": []
                }
            }
        else:
            total_plaques = sum(values)
            
            # Create custom labels with count and percentage
            labels = []
            percentages = []
            for i, (group, value) in enumerate(zip(groups, values)):
                percentage = (value / total_plaques) * 100
                percentages.append(percentage)
                labels.append(f"{group}<br>n={value} ({percentage:.1f}%)")
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                textfont=dict(size=14, family=font_family, color='black'),
                textinfo='label',
                textposition='inside',
                hole=0.3  # Creates a donut chart for better readability
            )])
            
            fig.update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size),
                showlegend=False,  # Remove legend since info is in labels
                margin=dict(t=50, b=50, l=50, r=50),
                # Add title
                title=dict(
                    text=f"Plaque Distribution (Total: {total_plaques})",
                    x=0.5,
                    font=dict(size=title_font_size, family=font_family)
                )
            )
            
            # Save JSON data for pie chart
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_number_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": True
                },
                "summary_statistics": {
                    "total_number": float(total_plaques),
                    "categories_present": len(groups)
                },
                "data": {
                    "categories": groups,
                    "values": [float(v) for v in values],
                    "percentages": [float(p) for p in percentages],
                    "colors": colors
                },
                "detailed_breakdown": {
                    "periventricular": {"count": float(peri_total), "percentage": float((peri_total/total_plaques)*100) if total_plaques > 0 else 0},
                    "paraventricular": {"count": float(para_total), "percentage": float((para_total/total_plaques)*100) if total_plaques > 0 else 0},
                    "juxtacortical": {"count": float(juxt_total), "percentage": float((juxt_total/total_plaques)*100) if total_plaques > 0 else 0}
                }
            }

    # Save pie chart image
    pio.write_image(fig, os.path.join(save_path, f'Pie_Plaque_Number_tp{tp}.png'),
                    width=600, height=600, scale=2)
    
    # Save pie chart data as JSON
    with open(os.path.join(save_path, f'pie_chart_number_distribution_data_tp{tp}.json'), 'w') as f:
        json.dump(pie_chart_data, f, indent=2)

    # ==================== DATA PROCESSING FOR OUTPUT ====================
    if len(num_peri) == 0 or len(num_para) == 0 or len(num_juxt) == 0:
        num_peri_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        num_para_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        num_juxt_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
    else:
        num_peri_ss = np.zeros_like(num_peri)
        num_para_ss = np.zeros_like(num_peri)
        num_juxt_ss = np.zeros_like(num_peri)

        num_peri_ss[0:len(num_peri)] = num_peri
        num_para_ss[0:len(num_para)] = num_para
        num_juxt_ss[0:len(num_juxt)] = num_juxt

    # ==================== SAVE COMPREHENSIVE SUMMARY JSON ====================
    # Create a comprehensive summary file with all key data
    comprehensive_summary = {
        "metadata": {
            "subject_id": id,
            "timepoint": tp,
            "analysis_type": "wmh_number_analysis",
            "timestamp": datetime.now().isoformat(),
            "generated_plots": [
                f"Slices_Plaque_Number_tp{tp}.png", 
                f"Category_Plaque_Number_tp{tp}.png",
                f"Pie_Plaque_Number_tp{tp}.png",
            ],
            "generated_data_files": [
                f"slice_wise_number_scatter_data_tp{tp}.json",
                f"category_number_grouped_bar_data_tp{tp}.json", 
                f"pie_chart_number_distribution_data_tp{tp}.json",
                f"comprehensive_number_summary_tp{tp}.json"
            ]
        },
        "global_statistics": {
            "total_plaques": len(wmh_num) if len(wmh_num) > 0 else 0,
            "area_statistics": {
                "mean": float(np.mean(wmh_num)) if len(wmh_num) > 0 else 0,
                "median": float(np.median(wmh_num)) if len(wmh_num) > 0 else 0,
                "std": float(np.std(wmh_num)) if len(wmh_num) > 0 else 0,
                "min": float(np.min(wmh_num)) if len(wmh_num) > 0 else 0,
                "max": float(np.max(wmh_num)) if len(wmh_num) > 0 else 0
            }
        },
        "category_breakdown": {
            "periventricular": {
                "count": int(np.sum(np.array(wmh_code) == 1)) if len(wmh_code) > 0 else 0,
                "total_count": float(np.sum(num_peri_ss)),
                "percentage_of_total": float((np.sum(num_peri_ss) / total_value * 100)) if total_value > 0 else 0
            },
            "paraventricular": {
                "count": int(np.sum(np.array(wmh_code) == 2)) if len(wmh_code) > 0 else 0,
                "total_count": float(np.sum(num_para_ss)),
                "percentage_of_total": float((np.sum(num_para_ss) / total_value * 100)) if total_value > 0 else 0
            },
            "juxtacortical": {
                "count": int(np.sum(np.array(wmh_code) == 3)) if len(wmh_code) > 0 else 0,
                "total_count": float(np.sum(num_juxt_ss)),
                "percentage_of_total": float((np.sum(num_juxt_ss) / total_value * 100)) if total_value > 0 else 0
            }
        },
        "slice_number_statistics": {
            "total_slices_with_plaques": int(np.sum(np.array(wmh_num) > 0)) if len(wmh_num) > 0 else 0,
            "max_plaques_per_slice": int(np.max(wmh_num)) if len(wmh_num) > 0 else 0,
            "mean_plaques_per_slice": float(np.mean(wmh_num)) if len(wmh_num) > 0 else 0.0,
            "std_plaques_per_slice": float(np.std(wmh_num)) if len(wmh_num) > 0 else 0.0,
        },
        "raw_data": {
            "wmh_codes": wmh_code if len(wmh_code) > 0 else [],
            "wmh_num_per_slice": wmh_num if len(wmh_num) > 0 else [],
            "category_numbers_per_slice": {
                "periventricular": num_peri_ss.tolist(),
                "paraventricular": num_para_ss.tolist(),
                "juxtacortical": num_juxt_ss.tolist()
            }
        }
    }
    
    # Save comprehensive summary
    with open(os.path.join(save_path, f'comprehensive_number_summary_tp{tp}.json'), 'w') as f:
        json.dump(comprehensive_summary, f, indent=2)

    # Update metadata (assuming these functions exist in your codebase)
    counts_general_data = {
        "whole_brain_total_count": int(total_value)
    }
    counts_periventricular_data = {
        "whole_brain_total_count": int(np.sum(num_peri_ss))
    }
    counts_paraventricular_data = {
        "whole_brain_total_count": int(np.sum(num_para_ss))
    }
    counts_juxtacortical_data = {
        "whole_brain_total_count": int(np.sum(num_juxt_ss))
    }

    # Note: These function calls will need to be uncommented if the functions exist
    update_wmh_data(wmh_m_data, "counts", "general", counts_general_data)
    update_wmh_data(wmh_m_data, "counts", "periventricular", counts_periventricular_data)
    update_wmh_data(wmh_m_data, "counts", "paraventricular", counts_paraventricular_data)
    update_wmh_data(wmh_m_data, "counts", "juxtacortical", counts_juxtacortical_data)

    return [num_peri_ss, num_para_ss, num_juxt_ss]

# Additional utility function for consistent styling across all plots
def set_publication_style():
    """
    Set global Plotly template for publication-ready figures
    """
    pio.templates["publication"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Arial", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            colorway=['#2E86AB', '#A23B72', '#F18F01', '#2F4858', '#87BBA2'],
            xaxis=dict(
                linecolor='black',
                linewidth=1,
                mirror=True,
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                linecolor='black',
                linewidth=1,
                mirror=True,
                gridcolor='lightgray',
                gridwidth=0.5
            )
        )
    )
    pio.templates.default = "publication"

# Call this function to set the publication style globally
# set_publication_style()

#
def area_show_subj(save_path, wmh_area, wmh_code, wmh_num, id, tp=0):
    """
    Enhanced function for publication-ready WMH lesion area visualization with JSON data export
    """
    
    if len(wmh_area) == 0:
        total_value = int(0)
        print(f"There is no seen plaque.")
    else:
        wmh_area = np.round(np.array(wmh_area), 1)
        total_value = int(np.round(np.sum(wmh_area)))
        total_value_v = int(np.round((np.sum(wmh_area) * voxel_size[-1] / 1000)))
        print(f"Total Area of Plaques: {total_value} mm²")
        print(f"Total Estimated Volume of Plaques: {total_value_v} cc")

    # ==================== SORTED AREAS SCATTER PLOT ====================
    if len(wmh_area) == 0:
        fig = go.Figure().update_layout(
            height=600, width=1100,
            xaxis_title='Plaque Index (Sorted by Area)',
            yaxis_title='Plaque Area (mm²)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for sorted areas scatter plot
        scatter_sorted_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "sorted_areas_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_plaques": 0,
                "total_area_mm2": 0,
                "total_volume_cc": 0
            },
            "data": {
                "plaque_areas": [],
                "plaque_codes": [],
                "sorted_indices": [],
                "x_values": [],
                "y_values": []
            },
            "categories": {
                "periventricular": {"count": 0, "areas": []},
                "paraventricular": {"count": 0, "areas": []},
                "juxtacortical": {"count": 0, "areas": []}
            }
        }
    else:
        # Sort areas and corresponding codes together
        sorted_indices = np.argsort(wmh_area)
        sorted_areas = wmh_area[sorted_indices]
        sorted_codes = np.array(wmh_code)[sorted_indices]
        x_values = np.arange(1, len(wmh_area) + 1)
        
        fig = go.Figure()
        
        # Create separate traces for each category for better legend control
        categories = {
            1: {'name': 'Periventricular', 'color': color_codes['peri']},
            2: {'name': 'Paraventricular', 'color': color_codes['para']},
            3: {'name': 'Juxtacortical', 'color': color_codes['juxt']}
        }
        
        # Add traces for each category
        for code, cat_info in categories.items():
            mask = sorted_codes == code
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=x_values[mask], 
                    y=sorted_areas[mask],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cat_info['color'],
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    name=cat_info['name'],
                    legendgroup=cat_info['name']
                ))
        
        fig.update_layout(
            height=600, width=1100,
            xaxis_title='Plaque Index (Sorted by Area)',
            yaxis_title='Plaque Area (mm²)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            )
        )
        
        # Save JSON data for sorted areas scatter plot
        scatter_sorted_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "sorted_areas_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_plaques": len(wmh_area),
                "total_area_mm2": float(total_value),
                "total_volume_cc": float(total_value_v) if len(wmh_area) > 0 else 0,
                "mean_area_mm2": float(np.mean(wmh_area)),
                "median_area_mm2": float(np.median(wmh_area)),
                "std_area_mm2": float(np.std(wmh_area)),
                "min_area_mm2": float(np.min(wmh_area)),
                "max_area_mm2": float(np.max(wmh_area))
            },
            "data": {
                "original_areas": wmh_area.tolist(),
                "original_codes": np.array(wmh_code).tolist(),
                "sorted_indices": sorted_indices.tolist(),
                "sorted_areas": sorted_areas.tolist(),
                "sorted_codes": sorted_codes.tolist(),
                "x_values": x_values.tolist(),
                "y_values": sorted_areas.tolist()
            },
            "categories": {
                "periventricular": {
                    "count": int(np.sum(np.array(wmh_code) == 1)),
                    "areas": wmh_area[np.array(wmh_code) == 1].tolist(),
                    "total_area_mm2": float(np.sum(wmh_area[np.array(wmh_code) == 1]))
                },
                "paraventricular": {
                    "count": int(np.sum(np.array(wmh_code) == 2)),
                    "areas": wmh_area[np.array(wmh_code) == 2].tolist(),
                    "total_area_mm2": float(np.sum(wmh_area[np.array(wmh_code) == 2]))
                },
                "juxtacortical": {
                    "count": int(np.sum(np.array(wmh_code) == 3)),
                    "areas": wmh_area[np.array(wmh_code) == 3].tolist(),
                    "total_area_mm2": float(np.sum(wmh_area[np.array(wmh_code) == 3]))
                }
            }
        }

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'All_Plaque_Area_tp{tp}.png'),
                    width=1200, height=600, scale=2)
    
    with open(os.path.join(save_path, f'sorted_areas_scatter_data_tp{tp}.json'), 'w') as f:
        json.dump(scatter_sorted_data, f, indent=2)

    # ==================== CALCULATE AREA PER SLICE ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        wmh_area_slice = []
    else:
        wmh_area_slice = np.zeros((len(wmh_num)))
        i = 0
        c = 0
        for k in wmh_num:
            if k == 0:
                c += 1
                continue
            wmh_area_slice[c] = np.sum(wmh_area[i:i + k])
            c += 1
            i += k

    # ==================== SLICE-WISE AREA SCATTER PLOT ====================
    if len(wmh_area_slice) == 0:
        fig = go.Figure().update_layout(
            height=600, width=1100,
            xaxis_title='Slice Number',
            yaxis_title='Total Plaque Area per Slice (mm²)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for slice-wise scatter plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_area_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
                "slices_with_plaques": 0,
                "total_area_mm2": 0
            },
            "data": {
                "slice_numbers": [],
                "area_per_slice": [],
                "plaques_per_slice": []
            }
        }

    else:
        x_values = np.arange(1, len(wmh_area_slice) + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=wmh_area_slice,
            mode='markers',
            marker=dict(
                size=8,
                color=color_codes['total'],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name='Area per Slice'
        ))
        
        fig.update_layout(
            height=600, width=1100,
            xaxis_title='Slice Number',
            yaxis_title='Total Plaque Area per Slice (mm²)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            showlegend=False
        )
        
        fig.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Area of Plaques: {total_value} mm²",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))
        
        # Save JSON data for slice-wise scatter plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_area_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_area_slice),
                "slices_with_plaques": int(np.sum(wmh_area_slice > 0)),
                "total_area_mm2": float(np.sum(wmh_area_slice)),
                "mean_area_per_slice_mm2": float(np.mean(wmh_area_slice)),
                "median_area_per_slice_mm2": float(np.median(wmh_area_slice)),
                "std_area_per_slice_mm2": float(np.std(wmh_area_slice)),
                "max_area_per_slice_mm2": float(np.max(wmh_area_slice))
            },
            "data": {
                "slice_numbers": x_values.tolist(),
                "area_per_slice": wmh_area_slice.tolist(),
                "plaques_per_slice": wmh_num
            }
        }

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'Slices_Plaque_Area_tp{tp}.png'),
                    width=1100, height=600, scale=2)
    
    with open(os.path.join(save_path, f'slice_wise_area_scatter_data_tp{tp}.json'), 'w') as f:
        json.dump(scatter_slices_data, f, indent=2)

    # ==================== CALCULATE CATEGORY-SPECIFIC AREAS ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        peri_ar = []
        para_ar = []
        juxt_ar = []
        area_peri_s = []
        area_para_s = []
        area_juxt_s = []
    else:
        peri_ar = np.where(np.array(wmh_code) == 1, 1, 0) * wmh_area
        para_ar = np.where(np.array(wmh_code) == 2, 1, 0) * wmh_area
        juxt_ar = np.where(np.array(wmh_code) == 3, 1, 0) * wmh_area

        area_peri_s = np.zeros_like(np.array(wmh_area_slice))
        area_para_s = np.zeros_like(np.array(wmh_area_slice))
        area_juxt_s = np.zeros_like(np.array(wmh_area_slice))

        i = 0
        c = 0
        for k in wmh_num:
            if k == 0:
                c += 1
                continue
            area_peri_s[c] = np.sum(peri_ar[i:i + k])
            area_para_s[c] = np.sum(para_ar[i:i + k])
            area_juxt_s[c] = np.sum(juxt_ar[i:i + k])
            c += 1
            i += k

    # ==================== GROUPED BAR PLOT FOR AREAS ====================
    nothing_to_show = False
    if len(wmh_num) == 0:
        nothing_to_show = True

    elif len(area_peri_s) == 0 or len(area_para_s) == 0 or len(area_juxt_s) == 0:
        nothing_to_show = True

    if nothing_to_show:
        plot = go.Figure().update_layout(
            height=600, width=1200,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        plot.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
                "periventricular_total": 0,
                "paraventricular_total": 0,
                "juxtacortical_total": 0
            },
            "data": {
                "slice_numbers": [],
                "periventricular_areas": [],
                "paraventricular_areas": [],
                "juxtacortical_areas": []
            }
        }
        
    else:
        x = list(range(1, len(wmh_num) + 1))
        
        plot = go.Figure()
        
        # Create grouped bars with offset positions
        bar_width = 0.25
        
        plot.add_trace(go.Bar(
            name='Periventricular',
            x=[xi - bar_width for xi in x], 
            y=list(area_peri_s),
            marker=dict(color=color_codes['peri']),
            width=bar_width,
            offsetgroup=1
        ))
        
        plot.add_trace(go.Bar(
            name='Paraventricular',
            x=x, 
            y=list(area_para_s),
            marker=dict(color=color_codes['para']),
            width=bar_width,
            offsetgroup=2
        ))
        
        plot.add_trace(go.Bar(
            name='Juxtacortical',
            x=[xi + bar_width for xi in x], 
            y=list(area_juxt_s),
            marker=dict(color=color_codes['juxt']),
            width=bar_width,
            offsetgroup=3
        ))
        
        plot.update_layout(
            barmode='group',
            height=600, width=1200,
            xaxis_title='Slice Number',
            yaxis_title='Total Plaque Area per Slice (mm²)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5,
                tick0=1,
                dtick=1  # Ensure integer ticks for slice numbers
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            ),
            bargap=0.2,  # Gap between groups of bars
            bargroupgap=0.1  # Gap between bars within a group
        )
        
        # Save JSON data for grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_num),
                "periventricular_total": float(np.sum(area_peri_s)),
                "paraventricular_total": float(np.sum(area_para_s)),
                "juxtacortical_total": float(np.sum(area_juxt_s)),
                "slices_with_peri": int(np.sum(area_peri_s > 0)),
                "slices_with_para": int(np.sum(area_para_s > 0)),
                "slices_with_juxt": int(np.sum(area_juxt_s > 0))
            },
            "data": {
                "slice_numbers": x,
                "periventricular_areas": area_peri_s.tolist(),
                "paraventricular_areas": area_para_s.tolist(),
                "juxtacortical_areas": area_juxt_s.tolist(),
                "bar_positions": {
                    "periventricular_x": [xi - bar_width for xi in x],
                    "paraventricular_x": x,
                    "juxtacortical_x": [xi + bar_width for xi in x]
                }
            }
        }
        
    # Save figure and JSON data
    pio.write_image(plot, os.path.join(save_path, f'Category_Plaque_Area_tp{tp}.png'),
                    width=1200, height=600, scale=2)
    
    with open(os.path.join(save_path, f'category_area_grouped_bar_data_tp{tp}.json'), 'w') as f:
        json.dump(grouped_bar_data, f, indent=2)

    # ==================== PIE CHART FOR AREAS ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0 or (np.sum(peri_ar) == 0 and np.sum(para_ar) == 0 and np.sum(juxt_ar) == 0):
        fig = go.Figure().update_layout(
            height=600, width=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No data to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for pie chart
        pie_chart_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "pie_chart_area_distribution",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_area_mm2": 0,
                "categories_present": 0
            },
            "data": {
                "categories": [],
                "values": [],
                "percentages": [],
                "colors": []
            }
        }

    else:
        # Calculate totals for each category
        peri_total = np.sum(peri_ar) if len(peri_ar) > 0 else 0
        para_total = np.sum(para_ar) if len(para_ar) > 0 else 0
        juxt_total = np.sum(juxt_ar) if len(juxt_ar) > 0 else 0
        
        # Only include categories with non-zero values
        groups = []
        values = []
        colors = []
        
        if peri_total > 0:
            groups.append('Periventricular')
            values.append(peri_total)
            colors.append(color_codes['peri'])
            
        if para_total > 0:
            groups.append('Paraventricular')
            values.append(para_total)
            colors.append(color_codes['para'])
            
        if juxt_total > 0:
            groups.append('Juxtacortical')
            values.append(juxt_total)
            colors.append(color_codes['juxt'])
        
        if len(groups) == 0:
            fig = go.Figure().update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size)
            )
            fig.add_annotation(dict(
                font=dict(color='gray', size=18, family=font_family),
                x=0.5, y=0.5, showarrow=False,
                text="No data to display", textangle=0,
                xanchor='center', xref="paper", yref="paper"
            ))
            
            # Save JSON data for pie chart (no data case)
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_area_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": False
                },
                "summary_statistics": {
                    "total_area_mm2": 0,
                    "categories_present": 0
                },
                "data": {
                    "categories": [],
                    "values": [],
                    "percentages": [],
                    "colors": []
                }
            }
        else:
            total_area = sum(values)
            percentages = [(value / total_area) * 100 for value in values]
            
            # Create custom labels with area and percentage
            labels = []
            for i, (group, value) in enumerate(zip(groups, values)):
                percentage = percentages[i]
                labels.append(f"{group}<br>{value:.1f} mm² ({percentage:.1f}%)")
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                textfont=dict(size=14, family=font_family, color='black'),
                textinfo='label',
                textposition='inside',
                hole=0.3
            )])
            
            fig.update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size),
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50),
                title=dict(
                    text=f"Area Distribution (Total: {total_area:.1f} mm²)",
                    x=0.5,
                    font=dict(size=title_font_size, family=font_family)
                )
            )
            
            # Save JSON data for pie chart
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_area_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": True
                },
                "summary_statistics": {
                    "total_area_mm2": float(total_area),
                    "categories_present": len(groups)
                },
                "data": {
                    "categories": groups,
                    "values": [float(v) for v in values],
                    "percentages": [float(p) for p in percentages],
                    "colors": colors
                },
                "detailed_breakdown": {
                    "periventricular": {"area_mm2": float(peri_total), "percentage": float((peri_total/total_area)*100) if total_area > 0 else 0},
                    "paraventricular": {"area_mm2": float(para_total), "percentage": float((para_total/total_area)*100) if total_area > 0 else 0},
                    "juxtacortical": {"area_mm2": float(juxt_total), "percentage": float((juxt_total/total_area)*100) if total_area > 0 else 0}
                }
            }

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'Pie_Plaque_Area_tp{tp}.png'),
                    width=600, height=600, scale=2)
    
    with open(os.path.join(save_path, f'pie_chart_area_distribution_data_tp{tp}.json'), 'w') as f:
        json.dump(pie_chart_data, f, indent=2)

    # ==================== BOX-WHISKER PLOT FOR AREA DISTRIBUTION ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0 or len(wmh_area) == 0:
        fig = go.Figure().update_layout(
            height=600, width=800,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for box plot
        box_plot_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "box_plot_area_distribution",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_plaques": 0,
                "categories_with_data": []
            },
            "data": {
                "periventricular": {"areas": [], "statistics": {}},
                "paraventricular": {"areas": [], "statistics": {}},
                "juxtacortical": {"areas": [], "statistics": {}}
            }
        }
    else:
        # Prepare data for box plots - separate areas by category
        peri_areas = wmh_area[np.array(wmh_code) == 1] if np.any(np.array(wmh_code) == 1) else []
        para_areas = wmh_area[np.array(wmh_code) == 2] if np.any(np.array(wmh_code) == 2) else []
        juxt_areas = wmh_area[np.array(wmh_code) == 3] if np.any(np.array(wmh_code) == 3) else []
        
        fig = go.Figure()
        
        # Add box plots for each category (only if data exists)
        if len(peri_areas) > 0:
            fig.add_trace(go.Box(
                y=peri_areas,
                name='Periventricular',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['peri'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',  # Show Weighted Average outliers
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        if len(para_areas) > 0:
            fig.add_trace(go.Box(
                y=para_areas,
                name='Paraventricular',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['para'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        if len(juxt_areas) > 0:
            fig.add_trace(go.Box(
                y=juxt_areas,
                name='Juxtacortical',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['juxt'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        # If no data for any category, show empty message
        if len(peri_areas) == 0 and len(para_areas) == 0 and len(juxt_areas) == 0:
            fig.add_annotation(dict(
                font=dict(color='gray', size=18, family=font_family),
                x=0.5, y=0.5, showarrow=False,
                text="No plaques to display", textangle=0,
                xanchor='center', xref="paper", yref="paper"
            ))
            
            # Save JSON data for box plot (no data case)
            box_plot_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "box_plot_area_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": False
                },
                "summary_statistics": {
                    "total_plaques": 0,
                    "categories_with_data": []
                },
                "data": {
                    "periventricular": {"areas": [], "statistics": {}},
                    "paraventricular": {"areas": [], "statistics": {}},
                    "juxtacortical": {"areas": [], "statistics": {}}
                }
            }
        else:
            # Calculate detailed statistics for each category
            def calculate_box_stats(areas):
                if len(areas) == 0:
                    return {}
                areas_array = np.array(areas)
                return {
                    "count": len(areas),
                    "mean": float(np.mean(areas_array)),
                    "median": float(np.median(areas_array)),
                    "std": float(np.std(areas_array)),
                    "min": float(np.min(areas_array)),
                    "max": float(np.max(areas_array)),
                    "q1": float(np.percentile(areas_array, 25)),
                    "q3": float(np.percentile(areas_array, 75)),
                    "iqr": float(np.percentile(areas_array, 75) - np.percentile(areas_array, 25))
                }
            
            categories_with_data = []
            if len(peri_areas) > 0: categories_with_data.append("periventricular")
            if len(para_areas) > 0: categories_with_data.append("paraventricular")
            if len(juxt_areas) > 0: categories_with_data.append("juxtacortical")
            
            # Save JSON data for box plot
            box_plot_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "box_plot_area_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": True
                },
                "summary_statistics": {
                    "total_plaques": len(wmh_area),
                    "categories_with_data": categories_with_data,
                    "total_area_mm2": float(total_value)
                },
                "data": {
                    "periventricular": {
                        "areas": peri_areas.tolist() if len(peri_areas) > 0 else [],
                        "statistics": calculate_box_stats(peri_areas)
                    },
                    "paraventricular": {
                        "areas": para_areas.tolist() if len(para_areas) > 0 else [],
                        "statistics": calculate_box_stats(para_areas)
                    },
                    "juxtacortical": {
                        "areas": juxt_areas.tolist() if len(juxt_areas) > 0 else [],
                        "statistics": calculate_box_stats(juxt_areas)
                    }
                }
            }
        
        fig.update_layout(
            height=600, width=800,
            xaxis_title='Plaque Categories',
            yaxis_title='Plaque Area (mm²)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        # Add summary statistics annotation
        total_plaques = len(wmh_area)
        fig.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size-2, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Plaques: {total_plaques} | Total Area: {total_value} mm²",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'BoxPlot_Plaque_Area_tp{tp}.png'),
                    width=600, height=600, scale=2)
    
    with open(os.path.join(save_path, f'box_plot_area_distribution_data_tp{tp}.json'), 'w') as f:
        json.dump(box_plot_data, f, indent=2)
    
    # ==================== DATA PROCESSING FOR OUTPUT ====================
    if len(wmh_num) == 0 or len(area_peri_s) == 0 or len(area_para_s) == 0 or len(area_juxt_s) == 0:
        area_peri_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        area_para_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        area_juxt_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
    else:
        area_peri_ss = np.zeros_like(area_peri_s)
        area_para_ss = np.zeros_like(area_peri_s)
        area_juxt_ss = np.zeros_like(area_peri_s)

        area_peri_ss[0:len(area_peri_s)] = area_peri_s
        area_para_ss[0:len(area_para_s)] = area_para_s
        area_juxt_ss[0:len(area_juxt_s)] = area_juxt_s

    # ==================== SAVE COMPREHENSIVE SUMMARY JSON ====================
    # Create a comprehensive summary file with all key data
    comprehensive_summary = {
        "metadata": {
            "subject_id": id,
            "timepoint": tp,
            "analysis_type": "wmh_area_analysis",
            "timestamp": datetime.now().isoformat(),
            "generated_plots": [
                f"All_Plaque_Area_tp{tp}.png",
                f"Slices_Plaque_Area_tp{tp}.png", 
                f"Category_Plaque_Area_tp{tp}.png",
                f"Pie_Plaque_Area_tp{tp}.png",
                f"BoxPlot_Plaque_Area_tp{tp}.png"
            ],
            "generated_data_files": [
                f"sorted_area_scatter_data_tp{tp}.json",
                f"slice_wise_area_scatter_data_tp{tp}.json",
                f"category_area_grouped_bar_data_tp{tp}.json", 
                f"pie_chart_area_distribution_data_tp{tp}.json",
                f"box_plot_area_distribution_data_tp{tp}.json",
                f"comprehensive_area_summary_tp{tp}.json"
            ]
        },
        "global_statistics": {
            "total_plaques": len(wmh_area) if len(wmh_area) > 0 else 0,
            "total_area_mm2": float(total_value),
            "total_volume_cc": float(total_value_v) if len(wmh_area) > 0 else 0,
            "area_statistics": {
                "mean_mm2": float(np.mean(wmh_area)) if len(wmh_area) > 0 else 0,
                "median_mm2": float(np.median(wmh_area)) if len(wmh_area) > 0 else 0,
                "std_mm2": float(np.std(wmh_area)) if len(wmh_area) > 0 else 0,
                "min_mm2": float(np.min(wmh_area)) if len(wmh_area) > 0 else 0,
                "max_mm2": float(np.max(wmh_area)) if len(wmh_area) > 0 else 0
            }
        },
        "category_breakdown": {
            "periventricular": {
                "count": int(np.sum(np.array(wmh_code) == 1)) if len(wmh_code) > 0 else 0,
                "total_area_mm2": float(np.sum(area_peri_ss)),
                "percentage_of_total": float((np.sum(area_peri_ss) / total_value * 100)) if total_value > 0 else 0
            },
            "paraventricular": {
                "count": int(np.sum(np.array(wmh_code) == 2)) if len(wmh_code) > 0 else 0,
                "total_area_mm2": float(np.sum(area_para_ss)),
                "percentage_of_total": float((np.sum(area_para_ss) / total_value * 100)) if total_value > 0 else 0
            },
            "juxtacortical": {
                "count": int(np.sum(np.array(wmh_code) == 3)) if len(wmh_code) > 0 else 0,
                "total_area_mm2": float(np.sum(area_juxt_ss)),
                "percentage_of_total": float((np.sum(area_juxt_ss) / total_value * 100)) if total_value > 0 else 0
            }
        },
        "slice_analysis": {
            "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
            "slices_with_plaques": int(np.sum(np.array(wmh_area_slice) > 0)) if len(wmh_area_slice) > 0 else 0,
            "slice_area_statistics": {
                "mean_area_per_slice_mm2": float(np.mean(wmh_area_slice)) if len(wmh_area_slice) > 0 else 0,
                "max_area_per_slice_mm2": float(np.max(wmh_area_slice)) if len(wmh_area_slice) > 0 else 0,
                "std_area_per_slice_mm2": float(np.std(wmh_area_slice)) if len(wmh_area_slice) > 0 else 0
            }
        },
        "raw_data": {
            "wmh_areas": wmh_area.tolist() if len(wmh_area) > 0 else [],
            "wmh_codes": wmh_code if len(wmh_code) > 0 else [],
            "wmh_num_per_slice": wmh_num if len(wmh_num) > 0 else [],
            "area_per_slice": wmh_area_slice.tolist() if len(wmh_area_slice) > 0 else [],
            "category_areas_per_slice": {
                "periventricular": area_peri_ss.tolist(),
                "paraventricular": area_para_ss.tolist(),
                "juxtacortical": area_juxt_ss.tolist()
            }
        }
    }
    
    # Save comprehensive summary
    with open(os.path.join(save_path, f'comprehensive_area_summary_tp{tp}.json'), 'w') as f:
        json.dump(comprehensive_summary, f, indent=2)

    # Update metadata
    area_general_data = {
        "area_of_all_found_WMH": list(np.uint16(wmh_area)) if len(wmh_area) > 0 else [],
        "whole_brain_total_area": int(total_value)
    }
    area_periventricular_data = {
        "whole_brain_total_area": int(np.sum(area_peri_ss))
    }
    area_paraventricular_data = {
        "whole_brain_total_area": int(np.sum(area_para_ss))
    }
    area_juxtacortical_data = {
        "whole_brain_total_area": int(np.sum(area_juxt_ss))
    }

    # Note: These function calls will need to be uncommented if the functions exist
    update_wmh_data(wmh_m_data, "area", "general", area_general_data)
    update_wmh_data(wmh_m_data, "area", "periventricular", area_periventricular_data)
    update_wmh_data(wmh_m_data, "area", "paraventricular", area_paraventricular_data)
    update_wmh_data(wmh_m_data, "area", "juxtacortical", area_juxtacortical_data)

    return [area_peri_ss, area_para_ss, area_juxt_ss]

#
def only_area_show_subj(save_path, a_mask, mask_name, color_code_, id, tp=0):
    """
    Enhanced function for publication-ready mask area visualization with JSON data export
    """
    # print('\n\n\n', a_mask.shape, '\n\n')
    area_mask = [np.round(np.sum(a_mask[..., i])*voxel_size[0]*voxel_size[1], 0) for i in range(a_mask.shape[-1])]
    total_value = np.round(np.sum(area_mask), 0)

    # ==================== CALCULATE CATEGORY-SPECIFIC AREAS ================
    # ==================== GROUPED BAR PLOT FOR AREAS =======================
    if len(area_mask) == 0:
        plot = go.Figure().update_layout(
            height=600, width=1200,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        plot.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No masks to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(area_mask) if len(area_mask) > 0 else 0,
                f"{mask_name}_area_total": 0,
            },
            "data": {
                "slice_numbers": [],
                f"{mask_name}_area": 0,
            }
        }
    else:
        x = list(range(1, len(area_mask) + 1))
        
        plot = go.Figure()
        
        # Create grouped bars with offset positions
        bar_width = 0.5
        
        plot.add_trace(go.Bar(
            name=mask_name,
            x=[xi for xi in x], 
            y=area_mask,
            marker=dict(color=color_code_),
            width=bar_width,
        ))
        
        plot.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Area of {mask_name}: {total_value} mm²",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))
        
        plot.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size, family=font_family),
            x=0.02, y=0.88, showarrow=False,
            text=f"Estimated Total Volume: {int(total_value*voxel_size[-1]/1000)} cc",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))

        plot.update_layout(
            barmode='stack',
            height=600, width=1200,
            xaxis_title='Slice Number',
            yaxis_title=f'Total {mask_name} Area per Slice (mm²)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            )
        )
        
        # Save JSON data for grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(area_mask) if len(area_mask) > 0 else 0,
                f"{mask_name}_area_total": float(np.sum(area_mask)),
            },
            "data": {
                "slice_numbers": x,
                f"{mask_name}_area": area_mask,
            },
            "area_statistics": {
                "mean_mm2": float(np.mean(area_mask)) if len(area_mask) > 0 else 0,
                "median_mm2": float(np.median(area_mask)) if len(area_mask) > 0 else 0,
                "std_mm2": float(np.std(area_mask)) if len(area_mask) > 0 else 0,
                "min_mm2": float(np.min(area_mask)) if len(area_mask) > 0 else 0,
                "max_mm2": float(np.max(area_mask)) if len(area_mask) > 0 else 0
            }
        }

    pio.write_image(plot, os.path.join(save_path, f'Category_{mask_name}_Area_tp{tp}.png'),
                    width=1200, height=600, scale=2)

    with open(os.path.join(save_path, f'category_{mask_name}_area_grouped_bar_data_tp{tp}.json'), 'w') as f:
        json.dump(grouped_bar_data, f, indent=2)

    # ==================== DATA PROCESSING FOR OUTPUT ====================
    if len(area_mask) == 0:
        area_mask_ss = np.zeros((len(a_mask))) if len(a_mask) > 0 else np.zeros(1)
        # area_para_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
    else:
        area_mask_ss = np.zeros_like(area_mask)
        # area_para_ss = np.zeros_like(area_peri_s)

        area_mask_ss[0:len(area_mask)] = area_mask
        # area_juxt_ss[0:len(area_juxt_s)] = area_juxt_s

    """    # Update metadata
    area_general_data = {
        f"area_of_all_found_{mask_name}": list(np.uint16(wmh_area)) if len(wmh_area) > 0 else [],
        "whole_brain_total_area": int(total_value)
    }
    area_periventricular_data = {
        "whole_brain_total_area": int(np.sum(area_peri_ss))
    }
    area_paraventricular_data = {
        "whole_brain_total_area": int(np.sum(area_para_ss))
    }
    area_juxtacortical_data = {
        "whole_brain_total_area": int(np.sum(area_juxt_ss))
    }"""

    return [area_mask_ss]

#
def int_show_subj(save_path, wmh_int, wmh_area, wmh_code, wmh_num, id, tp=0):
    """
    Enhanced function for publication-ready WMH lesion intensity visualization
    with category-based color coding for scatter plots and JSON data export
    """

    if len(wmh_int) == 0:
        total_value = 0
        print(f"There is no seen plaque.")
    else:
        wmh_int = np.round(np.array(wmh_int), 5)
        wmh_int[np.isnan(wmh_int)] = 0
        wmh_area = np.round(np.array(wmh_area), 1)
        
        # Calculate weighted average intensity
        total_value = np.round(100 * np.sum(wmh_int * wmh_area) / max(np.sum(wmh_area), 1), 1)
        print(f"Total Intensity Index of Plaques (weighted average): {total_value}%")

    # ==================== SORTED INTENSITIES SCATTER PLOT WITH CATEGORY COLORS ====================
    if len(wmh_int) == 0:
        fig = go.Figure().update_layout(
            height=600, width=1100,
            xaxis_title='Plaque Index (Sorted by Intensity)',
            yaxis_title='Plaque Intensity (%)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for empty sorted intensities plot
        scatter_sorted_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "sorted_intensities_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_plaques": 0,
                "total_intensities_index": 0.0,
            },
            "data": {
                "plaque_intensities": [],
                "plaque_codes": [],
                "sorted_indices": [],
                "x_values": [],
                "y_values": []
            },
            "categories": {
                "periventricular": {"count": 0, "intensities": []},
                "paraventricular": {"count": 0, "intensities": []},
                "juxtacortical": {"count": 0, "intensities": []}
            }
        }
    else:
        # Sort by intensity while keeping track of categories
        intensity_percent = 100 * wmh_int
        sorted_indices = np.argsort(intensity_percent)
        sorted_intensities = intensity_percent[sorted_indices]
        sorted_codes = np.array(wmh_code)[sorted_indices]
        x_values = np.arange(1, len(wmh_int) + 1)
        
        fig = go.Figure()
        
        # Create separate traces for each category to enable proper legend
        categories = {
            1: {'name': 'Periventricular', 'color': color_codes['peri']},
            2: {'name': 'Paraventricular', 'color': color_codes['para']},
            3: {'name': 'Juxtacortical', 'color': color_codes['juxt']}
        }

        # Add traces for each category
        for cat_code, cat_key in categories.items():
            mask = sorted_codes == cat_code
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=x_values[mask], 
                    y=sorted_intensities[mask],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cat_key['color'],
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    name=cat_key['name'],
                    legendgroup=cat_key['name']
                ))
        
        fig.update_layout(
            height=600, width=1100,
            xaxis_title='Plaque Index (Sorted by Intensity)',
            yaxis_title='Plaque Intensity (%)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            )
        )
        
        # Save JSON data for sorted scatter plot
        scatter_sorted_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "sorted_intensities_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_plaques": len(wmh_int),
                "total_intensities_index": float(total_value),
                "mean_intensities_index": float(np.mean(wmh_int)),
                "median_intensities_index": float(np.median(wmh_int)),
                "std_intensities_index": float(np.std(wmh_int)),
                "min_intensities_index": float(np.min(wmh_int)),
                "max_intensities_index": float(np.max(wmh_int))
            },
            "data": {
                "original_intensities": wmh_int.tolist(),
                "original_codes": np.array(wmh_code).tolist(),
                "sorted_indices": sorted_indices.tolist(),
                "sorted_intensities": sorted_intensities.tolist(),
                "sorted_codes": sorted_codes.tolist(),
                "x_values": x_values.tolist(),
                "y_values": sorted_intensities.tolist()
            },
            "categories": {
                "periventricular": {
                    "count": int(np.sum(np.array(wmh_code) == 1)),
                    "intensities": wmh_int[np.array(wmh_code) == 1].tolist(),
                    "total_intensities_index": float(np.sum(wmh_int[np.array(wmh_code) == 1]))
                },
                "paraventricular": {
                    "count": int(np.sum(np.array(wmh_code) == 2)),
                    "intensities": wmh_int[np.array(wmh_code) == 2].tolist(),
                    "total_intensities_index": float(np.sum(wmh_int[np.array(wmh_code) == 2]))
                },
                "juxtacortical": {
                    "count": int(np.sum(np.array(wmh_code) == 3)),
                    "intensities": wmh_int[np.array(wmh_code) == 3].tolist(),
                    "total_intensities_index": float(np.sum(wmh_int[np.array(wmh_code) == 3]))
                }
            }
        }

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'All_Plaque_Intensity_tp{tp}.png'),
                    width=1200, height=600, scale=2)

    with open(os.path.join(save_path, f'sorted_intensities_scatter_data_tp{tp}.json'), 'w') as f:
        json.dump(scatter_sorted_data, f, indent=2)

    # ==================== CALCULATE INTENSITY PER SLICE ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        wmh_int_slice = []
    else:
        wmh_int_slice = np.zeros((len(wmh_num)))
        i = 0
        c = 0
        for k in wmh_num:
            if k == 0:
                c += 1
                continue
            # Weighted average intensity per slice
            wmh_int_slice[c] = np.sum(wmh_int[i:i + k] * wmh_area[i:i + k]) / np.sum(wmh_area[i:i + k])
            c += 1
            i += k

    # ==================== SLICE-WISE INTENSITY SCATTER PLOT ====================
    if len(wmh_int_slice) == 0:
        fig = go.Figure().update_layout(
            height=600, width=1100,
            xaxis_title='Slice Number',
            yaxis_title='Average Plaque Intensity per Slice (%)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for empty slice-wise plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_intensity_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
                "slices_with_plaques": 0,
                "total_intensity_index": 0.0
            },
            "data": {
                "slice_numbers": [],
                "intensity_per_slice": [],
                "plaques_per_slice": []
            }
        }

    else:
        x_values = np.arange(1, len(wmh_int_slice) + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=100 * wmh_int_slice,
            mode='markers',
            marker=dict(
                size=8,
                color=color_codes['total'],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name='Intensity per Slice'
        ))
        
        fig.update_layout(
            height=600, width=1100,
            xaxis_title='Slice Number',
            yaxis_title='Weighted Average Plaque Intensity per Slice (%)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            showlegend=False
        )
        
        fig.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Intensity Index of Plaques: {total_value}%",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))
        
        # Save JSON data for slice-wise plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_intensity_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_int_slice),
                "slices_with_plaques": int(np.sum(100*wmh_int_slice > 0)),
                "total_intensity_index": float(np.sum(100*wmh_int_slice)),
                "mean_intensity_per_slice_%": float(np.mean(100*wmh_int_slice)),
                "median_intensity_per_slice_%": float(np.median(100*wmh_int_slice)),
                "std_intensity_per_slice_%": float(np.std(100*wmh_int_slice)),
                "max_intensity_per_slice_%": float(np.max(100*wmh_int_slice))
            },
            "data": {
                "slice_numbers": x_values.tolist(),
                "intensity_per_slice": (100*wmh_int_slice).tolist(),
                "plaques_per_slice": wmh_num
            }
        }
        
    # Save figure and slice-wise JSON data
    pio.write_image(fig, os.path.join(save_path, f'Slices_Plaque_Intensity_tp{tp}.png'),
                    width=1100, height=600, scale=2)

    with open(os.path.join(save_path, f'slice_wise_intensity_scatter_data_tp{tp}.json'), 'w') as f:
        json.dump(scatter_slices_data, f, indent=2)

    # ==================== CALCULATE CATEGORY-SPECIFIC INTENSITIES ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        peri_in = []
        para_in = []
        juxt_in = []
        int_peri_s = []
        int_para_s = []
        int_juxt_s = []
        int_peri_ss = []
        int_para_ss = []
        int_juxt_ss = []
        whole_area = 1
    else:
        peri_in = (np.where(np.array(wmh_code) == 1, 1, 0) * wmh_int)
        para_in = (np.where(np.array(wmh_code) == 2, 1, 0) * wmh_int)
        juxt_in = (np.where(np.array(wmh_code) == 3, 1, 0) * wmh_int)

        int_peri_s = np.zeros_like(np.array(wmh_int_slice))
        int_peri_ss = np.zeros_like(np.array(wmh_int_slice))
        int_para_s = np.zeros_like(np.array(wmh_int_slice))
        int_para_ss = np.zeros_like(np.array(wmh_int_slice))
        int_juxt_s = np.zeros_like(np.array(wmh_int_slice))
        int_juxt_ss = np.zeros_like(np.array(wmh_int_slice))
        
        i = 0
        c = 0
        whole_area = max(np.sum(wmh_area), 1)
        whole_area_peri = max(np.sum((np.where(np.array(wmh_code) == 1, 1, 0) * wmh_area)), 1)
        whole_area_para = max(np.sum((np.where(np.array(wmh_code) == 2, 1, 0) * wmh_area)), 1)
        whole_area_juxt = max(np.sum((np.where(np.array(wmh_code) == 3, 1, 0) * wmh_area)), 1)

        for k in wmh_num:
            if k == 0:
                c += 1
                continue
            # Calculate weighted intensities by category per slice
            int_peri_s[c] = np.sum(peri_in[i:i + k] * wmh_area[i:i + k])
            int_peri_ss[c] = np.sum(peri_in[i:i + k] * wmh_area[i:i + k]) / np.sum(np.where(peri_in[i:i + k]>0, 1, 0) * wmh_area[i:i + k]) if np.sum(np.where(peri_in[i:i + k]>0, 1, 0)) > 0 else 0
            int_para_s[c] = np.sum(para_in[i:i + k] * wmh_area[i:i + k])
            int_para_ss[c] = np.sum(para_in[i:i + k] * wmh_area[i:i + k]) / np.sum(np.where(para_in[i:i + k]>0, 1, 0) * wmh_area[i:i + k]) if np.sum(np.where(para_in[i:i + k]>0, 1, 0)) > 0 else 0
            int_juxt_s[c] = np.sum(juxt_in[i:i + k] * wmh_area[i:i + k])
            int_juxt_ss[c] = np.sum(juxt_in[i:i + k] * wmh_area[i:i + k]) / np.sum(np.where(juxt_in[i:i + k]>0, 1, 0) * wmh_area[i:i + k]) if np.sum(np.where(juxt_in[i:i + k]>0, 1, 0)) > 0 else 0
            c += 1
            i += k
            
    # ==================== GROUPED BAR PLOT FOR INTENSITIES ====================
    nothing_to_show = False
    if len(wmh_num) == 0:
        nothing_to_show = True

    elif len(int_peri_ss) == 0 or len(int_para_ss) == 0 or len(int_juxt_ss) == 0:
        nothing_to_show = True

    if nothing_to_show:
        plot = go.Figure().update_layout(
            height=600, width=1200,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        plot.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for empty grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
                "periventricular_total": 0,
                "paraventricular_total": 0,
                "juxtacortical_total": 0
            },
            "data": {
                "slice_numbers": [],
                "periventricular_intensities": [],
                "paraventricular_intensities": [],
                "juxtacortical_intensities": []
            }
        }

    else:
        x = list(range(1, len(wmh_num) + 1))
        
        plot = go.Figure()
        
        # Create grouped bars with offset positions
        bar_width = 0.25
        
        plot.add_trace(go.Bar(
            name='Periventricular',
            x=[xi - bar_width for xi in x], 
            y=list(100 * int_peri_ss),
            marker=dict(color=color_codes['peri']),
            width=bar_width,
            offsetgroup=1
        ))
        
        plot.add_trace(go.Bar(
            name='Paraventricular',
            x=x, 
            y=list(100 * int_para_ss),
            marker=dict(color=color_codes['para']),
            width=bar_width,
            offsetgroup=2
        ))
        
        plot.add_trace(go.Bar(
            name='Juxtacortical',
            x=[xi + bar_width for xi in x], 
            y=list(100 * int_juxt_ss),
            marker=dict(color=color_codes['juxt']),
            width=bar_width,
            offsetgroup=3
        ))
        
        plot.update_layout(
            barmode='group',  # Changed from 'stack' to 'group'
            height=600, width=1200,
            xaxis_title='Slice Number',
            yaxis_title='Weighted Average Plaque Intensity per Slice (%)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5,
                tick0=1,
                dtick=1  # Ensure integer ticks for slice numbers
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            ),
            bargap=0.2,  # Gap between groups of bars
            bargroupgap=0.1  # Gap between bars within a group
        )
        
        # Save JSON data for grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_num),
                "periventricular_total": float(np.sum(100 * int_peri_ss)),
                "paraventricular_total": float(np.sum(100 * int_para_ss)),
                "juxtacortical_total": float(np.sum(100 * int_juxt_ss)),
                "slices_with_peri": int(np.sum(100 * int_peri_ss > 0)),
                "slices_with_para": int(np.sum(100 * int_para_ss > 0)),
                "slices_with_juxt": int(np.sum(100 * int_juxt_ss > 0))
            },
            "data": {
                "slice_numbers": x,
                "periventricular_intensity": (100 * int_peri_ss).tolist(),
                "paraventricular_intensity": (100 * int_para_ss).tolist(),
                "juxtacortical_intensity": (100 * int_juxt_ss).tolist(),
                "bar_positions": {
                    "periventricular_x": [xi - bar_width for xi in x],
                    "paraventricular_x": x,
                    "juxtacortical_x": [xi + bar_width for xi in x]
                }
            }
        }

    # Save figure and JSON data
    pio.write_image(plot, os.path.join(save_path, f'Category_Plaque_Intensity_tp{tp}.png'),
                    width=1200, height=600, scale=2)

    with open(os.path.join(save_path, f'category_intensity_grouped_bar_data_tp{tp}.json'), 'w') as f:
        json.dump(grouped_bar_data, f, indent=2)

    # ==================== PIE CHART FOR INTENSITIES ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0 or (np.sum(peri_in) == 0 and np.sum(para_in) == 0 and np.sum(juxt_in) == 0):
        fig = go.Figure().update_layout(
            height=600, width=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No data to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for empty pie chart
        pie_chart_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "pie_chart_intensity_distribution",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_intensity_index": 0.0,
                "categories_present": 0
            },
            "data": {
                "categories": [],
                "values": [],
                "percentages": [],
                "colors": []
            }
        }

    else:
        # Calculate totals for each category
        peri_total = np.round(np.sum(int_peri_s) / whole_area * 100, 1)
        para_total = np.round(np.sum(int_para_s) / whole_area * 100, 1)
        juxt_total = np.round(np.sum(int_juxt_s) / whole_area * 100, 1)
        
        # Only include categories with non-zero values
        groups = []
        values = []
        colors = []
        
        if peri_total > 0:
            groups.append('Periventricular')
            values.append(peri_total)
            colors.append(color_codes['peri'])
            
        if para_total > 0:
            groups.append('Paraventricular')
            values.append(para_total)
            colors.append(color_codes['para'])
            
        if juxt_total > 0:
            groups.append('Juxtacortical')
            values.append(juxt_total)
            colors.append(color_codes['juxt'])
        
        if len(groups) == 0:
            fig = go.Figure().update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size)
            )
            fig.add_annotation(dict(
                font=dict(color='gray', size=18, family=font_family),
                x=0.5, y=0.5, showarrow=False,
                text="No data to display", textangle=0,
                xanchor='center', xref="paper", yref="paper"
            ))
            
            # Save JSON data for empty pie chart
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_intensity_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": False
                },
                "summary_statistics": {
                    "total_intensity_index": 0,
                    "categories_present": 0
                },
                "data": {
                    "categories": [],
                    "values": [],
                    "percentages": [],
                    "colors": []
                }
            }

        else:
            total_intensity = sum(values)
            percentages = []
            
            # Create custom labels with intensity and percentage
            labels = []
            for i, (group, value, cat_area) in enumerate(zip(groups, values, [whole_area_peri, whole_area_para, whole_area_juxt])):
                percentage = (value / total_intensity) * 100
                percentages.append(percentage)
                value = (value * whole_area) / cat_area
                labels.append(f"{group}<br>{value:.1f}% ({percentage:.1f}%)")
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                textfont=dict(size=14, family=font_family, color='black'),
                textinfo='label',
                textposition='inside',
                hole=0.3
            )])
            
            fig.update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size),
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50),
                title=dict(
                    text=f"Intensity Distribution (Total: {total_intensity:.1f}%)",
                    x=0.5,
                    font=dict(size=title_font_size, family=font_family)
                )
            )
            
            # Save JSON data for populated pie chart
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_intensity_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": True
                },
                "summary_statistics": {
                    "total_intensity_index": float(total_intensity),
                    "categories_present": len(groups)
                },
                "data": {
                    "categories": groups,
                    "values": [float(v) for v in values],
                    "percentages": [float(p) for p in percentages],
                    "colors": colors
                },
                "detailed_breakdown": {
                    "periventricular": {"intensity_%": float(peri_total), "percentage": float((peri_total/total_intensity)*100) if total_intensity > 0 else 0.0},
                    "paraventricular": {"intensity_%": float(para_total), "percentage": float((para_total/total_intensity)*100) if total_intensity > 0 else 0.0},
                    "juxtacortical": {"intensity_%": float(juxt_total), "percentage": float((juxt_total/total_intensity)*100) if total_intensity > 0 else 0.0}
                }
            }

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'Pie_Plaque_Intensity_tp{tp}.png'),
                    width=600, height=600, scale=2)

    with open(os.path.join(save_path, f'pie_chart_intensity_distribution_data_tp{tp}.json'), 'w') as f:
        json.dump(pie_chart_data, f, indent=2)

    # ==================== BOX-WHISKER PLOT FOR INTENSITY DISTRIBUTION ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0 or len(wmh_int) == 0:
        fig = go.Figure().update_layout(
            height=600, width=800,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for empty box plot
        box_plot_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "box_plot_intensity_distribution",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_plaques": 0,
                "categories_with_data": []
            },
            "data": {
                "periventricular": {"intensity": [], "statistics": {}},
                "paraventricular": {"intensity": [], "statistics": {}},
                "juxtacortical": {"intensity": [], "statistics": {}}
            }
        }

    else:
        # Prepare data for box plots - separate intensity by category
        peri_ints = 100 * (wmh_int[np.array(wmh_code) == 1] if np.any(np.array(wmh_code) == 1) else [])
        para_ints = 100 * (wmh_int[np.array(wmh_code) == 2] if np.any(np.array(wmh_code) == 2) else [])
        juxt_ints = 100 * (wmh_int[np.array(wmh_code) == 3] if np.any(np.array(wmh_code) == 3) else [])
        
        fig = go.Figure()
        
        # Add box plots for each category (only if data exists)
        if len(peri_ints) > 0:
            fig.add_trace(go.Box(
                y=peri_ints,
                name='Periventricular',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['peri'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',  # Show outliers
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        if len(para_ints) > 0:
            fig.add_trace(go.Box(
                y=para_ints,
                name='Paraventricular',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['para'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        if len(juxt_ints) > 0:
            fig.add_trace(go.Box(
                y=juxt_ints,
                name='Juxtacortical',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['juxt'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        # If no data for any category, show empty message
        if len(peri_ints) == 0 and len(para_ints) == 0 and len(juxt_ints) == 0:
            fig.add_annotation(dict(
                font=dict(color='gray', size=18, family=font_family),
                x=0.5, y=0.5, showarrow=False,
                text="No plaques to display", textangle=0,
                xanchor='center', xref="paper", yref="paper"
            ))

            # Save JSON data for box plot (no data case)
            box_plot_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "box_plot_intensity_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": False
                },
                "summary_statistics": {
                    "total_plaques": 0,
                    "categories_with_data": []
                },
                "data": {
                    "periventricular": {"intensity": [], "statistics": {}},
                    "paraventricular": {"intensity": [], "statistics": {}},
                    "juxtacortical": {"intensity": [], "statistics": {}}
                }
            }
        else:
            # Calculate detailed statistics for each category
            def calculate_box_stats(intensity):
                if len(intensity) == 0:
                    return {}
                intensity_array = np.array(intensity)
                return {
                    "count": len(intensity),
                    "mean": float(np.mean(intensity_array)),
                    "median": float(np.median(intensity_array)),
                    "std": float(np.std(intensity_array)),
                    "min": float(np.min(intensity_array)),
                    "max": float(np.max(intensity_array)),
                    "q1": float(np.percentile(intensity_array, 25)),
                    "q3": float(np.percentile(intensity_array, 75)),
                    "iqr": float(np.percentile(intensity_array, 75) - np.percentile(intensity_array, 25))
                }
            
            categories_with_data = []
            if len(peri_ints) > 0: categories_with_data.append("periventricular")
            if len(para_ints) > 0: categories_with_data.append("paraventricular")
            if len(juxt_ints) > 0: categories_with_data.append("juxtacortical")
            
            # Save JSON data for box plot
            box_plot_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "box_plot_intensity_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": True
                },
                "summary_statistics": {
                    "total_plaques": len(wmh_int),
                    "categories_with_data": categories_with_data,
                    "total_intensity_index": float(total_value)
                },
                "data": {
                    "periventricular": {
                        "intensity": peri_ints.tolist() if len(peri_ints) > 0 else [],
                        "statistics": calculate_box_stats(peri_ints)
                    },
                    "paraventricular": {
                        "intensity": para_ints.tolist() if len(para_ints) > 0 else [],
                        "statistics": calculate_box_stats(para_ints)
                    },
                    "juxtacortical": {
                        "intensity": juxt_ints.tolist() if len(juxt_ints) > 0 else [],
                        "statistics": calculate_box_stats(juxt_ints)
                    }
                }
            }

        fig.update_layout(
            height=600, width=800,
            xaxis_title='Plaque Categories',
            yaxis_title='Plaque Intensity (%)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        # Add summary statistics annotation
        total_plaques = len(wmh_int)
        fig.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size-2, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Plaques: {total_plaques} | Total Intensity: {total_value} %",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'BoxPlot_Plaque_Intensity_tp{tp}.png'),
                    width=600, height=600, scale=2)

    with open(os.path.join(save_path, f'box_plot_intensity_distribution_data_tp{tp}.json'), 'w') as f:
        json.dump(box_plot_data, f, indent=2)

    # ==================== DATA PROCESSING FOR OUTPUT ====================
    if len(int_peri_s) == 0 or len(int_para_s) == 0 or len(int_juxt_s) == 0:
        int_peri_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        int_para_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        int_juxt_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
    else:
        int_peri_ss = np.zeros_like(int_peri_s)
        int_para_ss = np.zeros_like(int_peri_s)
        int_juxt_ss = np.zeros_like(int_peri_s)

        int_peri_ss[0:len(int_peri_s)] = 100 * int_peri_s / whole_area
        int_para_ss[0:len(int_para_s)] = 100 * int_para_s / whole_area
        int_juxt_ss[0:len(int_juxt_s)] = 100 * int_juxt_s / whole_area

    # ==================== SAVE COMPREHENSIVE SUMMARY JSON ====================
    # Create a comprehensive summary file with all key data
    comprehensive_summary = {
        "metadata": {
            "subject_id": id,
            "timepoint": tp,
            "analysis_type": "wmh_intensity_analysis",
            "timestamp": datetime.now().isoformat(),
            "generated_plots": [
                f"All_Plaque_Intensity_tp{tp}.png",
                f"Slices_Plaque_Intensity_tp{tp}.png", 
                f"Category_Plaque_Intensity_tp{tp}.png",
                f"Pie_Plaque_Intensity_tp{tp}.png",
                f"BoxPlot_Plaque_Intensity_tp{tp}.png"
            ],
            "generated_data_files": [
                f"sorted_intensity_scatter_data_tp{tp}.json",
                f"slice_wise_intensity_scatter_data_tp{tp}.json",
                f"category_intensity_grouped_bar_data_tp{tp}.json", 
                f"pie_chart_intensity_distribution_data_tp{tp}.json",
                f"box_plot_intensity_distribution_data_tp{tp}.json",
                f"comprehensive_intensity_summary_tp{tp}.json"
            ]
        },
        "global_statistics": {
            "total_plaques": len(wmh_int) if len(wmh_int) > 0 else 0,
            "total_intensity_index": float(total_value),
            "intensity_statistics": {
                "mean_%": float(np.mean(wmh_int)) if len(wmh_int) > 0 else 0,
                "median_%": float(np.median(wmh_int)) if len(wmh_int) > 0 else 0,
                "std_%": float(np.std(wmh_int)) if len(wmh_int) > 0 else 0,
                "min_%": float(np.min(wmh_int)) if len(wmh_int) > 0 else 0,
                "max_%": float(np.max(wmh_int)) if len(wmh_int) > 0 else 0
            }
        },
        "category_breakdown": {
            "periventricular": {
                "count": int(np.sum(np.array(wmh_code) == 1)) if len(wmh_code) > 0 else 0.0,
                "total_intensity_index": float(np.sum(int_peri_ss)),
                "percentage_of_total": float((np.sum(int_peri_ss) / total_value * 100)) if total_value > 0 else 0.0
            },
            "paraventricular": {
                "count": int(np.sum(np.array(wmh_code) == 2)) if len(wmh_code) > 0 else 0.0,
                "total_intensity_index": float(np.sum(int_para_ss)),
                "percentage_of_total": float((np.sum(int_para_ss) / total_value * 100)) if total_value > 0 else 0.0
            },
            "juxtacortical": {
                "count": int(np.sum(np.array(wmh_code) == 3)) if len(wmh_code) > 0 else 0.0,
                "total_intensity_index": float(np.sum(int_juxt_ss)),
                "percentage_of_total": float((np.sum(int_juxt_ss) / total_value * 100)) if total_value > 0 else 0.0
            }
        },
        "slice_analysis": {
            "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
            "slices_with_plaques": int(np.sum(np.array(wmh_int_slice) > 0)) if len(wmh_int_slice) > 0 else 0,
            "slice_intensity_statistics": {
                "mean_intensity_per_slice_%": float(np.mean(wmh_int_slice)) if len(wmh_int_slice) > 0 else 0,
                "max_intensity_per_slice_%": float(np.max(wmh_int_slice)) if len(wmh_int_slice) > 0 else 0,
                "std_intensity_per_slice_%": float(np.std(wmh_int_slice)) if len(wmh_int_slice) > 0 else 0
            }
        },
        "raw_data": {
            "wmh_areas": wmh_area.tolist() if len(wmh_area) > 0 else [],
            "wmh_ints": wmh_int.tolist() if len(wmh_int) > 0 else [],
            "wmh_codes": wmh_code if len(wmh_code) > 0 else [],
            "wmh_num_per_slice": wmh_num if len(wmh_num) > 0 else [],
            "intensity_per_slice": wmh_int_slice.tolist() if len(wmh_int_slice) > 0 else [],
            "category_intensity_per_slice": {
                "periventricular": int_peri_ss.tolist(),
                "paraventricular": int_para_ss.tolist(),
                "juxtacortical": int_juxt_ss.tolist()
            }
        }
    }
    
    # Save comprehensive summary
    with open(os.path.join(save_path, f'comprehensive_intensity_summary_tp{tp}.json'), 'w') as f:
        json.dump(comprehensive_summary, f, indent=2)

    # Update metadata
    intensity_general_data = {
        "intensity_of_all_found_WMH": list(np.uint16(100 * wmh_int)) if len(wmh_int) > 0 else [],
        "whole_brain_total_intensity": int(total_value)
    }
    intensity_periventricular_data = {
        "whole_brain_total_intensity": int(np.sum(int_peri_ss))
    }
    intensity_paraventricular_data = {
        "whole_brain_total_intensity": int(np.sum(int_para_ss))
    }
    intensity_juxtacortical_data = {
        "whole_brain_total_intensity": int(np.sum(int_juxt_ss))
    }

    # Note: These function calls will need to be uncommented if the functions exist
    update_wmh_data(wmh_m_data, "intensity", "general", intensity_general_data)
    update_wmh_data(wmh_m_data, "intensity", "periventricular", intensity_periventricular_data)
    update_wmh_data(wmh_m_data, "intensity", "paraventricular", intensity_paraventricular_data)
    update_wmh_data(wmh_m_data, "intensity", "juxtacortical", intensity_juxtacortical_data)

    return [int_peri_ss, int_para_ss, int_juxt_ss]

#
def Cdist_show_subj(save_path, wmh_cob, wmh_area, wmh_code, wmh_num, id, tp=0):
    """
    Enhanced function for publication-ready WMH lesion penetration visualization
    Consistent with the area_show_subj function styling and structure
    Now includes JSON data export for each visualization
    """

    if len(wmh_cob) == 0:
        total_value = int(0)
        print(f"There is no seen plaque.")
    else:
        wmh_cob = np.array(wmh_cob)
        wmh_area = np.round(np.array(wmh_area), 1)
        wmh_code = np.array(wmh_code)
        
        # Calculate weighted average penetration
        total_value = int(np.round(np.sum(wmh_cob * wmh_area) / max(np.sum(wmh_area), 1)))
        print(f"Total Penetration of Plaques (weighted average): {total_value} mm")

    # ==================== SORTED PENETRATION SCATTER PLOT WITH CATEGORY COLORS ====================
    if len(wmh_cob) == 0:        
        fig = go.Figure().update_layout(
            height=600, width=1100,
            xaxis_title='Plaque Index (Sorted by Penetration)',
            yaxis_title='Plaque Penetration (mm)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for empty sorted penetration plot
        scatter_sorted_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "sorted_penetrations_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_plaques": 0,
                "total_penetrations_mm": 0.0,
            },
            "data": {
                "plaque_penetrations": [],
                "plaque_codes": [],
                "sorted_indices": [],
                "x_values": [],
                "y_values": []
            },
            "categories": {
                "periventricular": {"count": 0, "penetrations": []},
                "paraventricular": {"count": 0, "penetrations": []},
                "juxtacortical": {"count": 0, "penetrations": []}
            }
        }
    else:
        # Sort penetration values while keeping track of original indices
        sorted_indices = np.argsort(wmh_cob)
        sorted_penetration = wmh_cob[sorted_indices]
        sorted_codes = np.array(wmh_code)[sorted_indices]
        x_values = np.arange(1, len(wmh_cob) + 1)
    
        fig = go.Figure()
        
        # Create separate traces for each category to enable proper legend
        categories = {
            1: {'name': 'Periventricular', 'color': color_codes['peri']},
            2: {'name': 'Paraventricular', 'color': color_codes['para']},
            3: {'name': 'Juxtacortical', 'color': color_codes['juxt']}
        }

        # Add traces for each category
        for cat_code, cat_key in categories.items():
            mask = sorted_codes == cat_code
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=x_values[mask], 
                    y=sorted_penetration[mask],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cat_key['color'],
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    name=cat_key['name'],
                    legendgroup=cat_key['name']
                ))
        
        fig.update_layout(
            height=600, width=1100,
            xaxis_title='Plaque Index (Sorted by Penetration)',
            yaxis_title='Plaque Penetration (mm)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            )
        )

        # Save JSON data for sorted scatter plot
        scatter_sorted_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "sorted_penetrations_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_plaques": len(wmh_cob),
                "total_penetration_mm": float(total_value),
                "mean_penetration_mm": float(np.mean(wmh_cob)),
                "median_penetration_mm": float(np.median(wmh_cob)),
                "std_penetration_mm": float(np.std(wmh_cob)),
                "min_penetration_mm": float(np.min(wmh_cob)),
                "max_penetration_mm": float(np.max(wmh_cob))
            },
            "data": {
                "original_penetration": wmh_cob.tolist(),
                "original_codes": np.array(wmh_code).tolist(),
                "sorted_indices": sorted_indices.tolist(),
                "sorted_penetration": sorted_penetration.tolist(),
                "sorted_codes": sorted_codes.tolist(),
                "x_values": x_values.tolist(),
                "y_values": sorted_penetration.tolist()
            },
            "categories": {
                "periventricular": {
                    "count": int(np.sum(np.array(wmh_code) == 1)),
                    "penetration": wmh_cob[np.array(wmh_code) == 1].tolist(),
                    "total_penetration_mm": float(np.sum(wmh_cob[np.array(wmh_code) == 1]))
                },
                "paraventricular": {
                    "count": int(np.sum(np.array(wmh_code) == 2)),
                    "penetration": wmh_cob[np.array(wmh_code) == 2].tolist(),
                    "total_penetration_mm": float(np.sum(wmh_cob[np.array(wmh_code) == 2]))
                },
                "juxtacortical": {
                    "count": int(np.sum(np.array(wmh_code) == 3)),
                    "penetration": wmh_cob[np.array(wmh_code) == 3].tolist(),
                    "total_penetration_mm": float(np.sum(wmh_cob[np.array(wmh_code) == 3]))
                }
            }
        }

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'All_Plaque_Penetration_tp{tp}.png'),
                    width=1200, height=600, scale=2)

    with open(os.path.join(save_path, f'sorted_penetration_scatter_data_tp{tp}.json'), 'w') as f:
        json.dump(scatter_sorted_data, f, indent=2)

    # ==================== CALCULATE PENETRATION PER SLICE ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        wmh_cob_slice = []
    else:
        wmh_cob_slice = np.zeros((len(wmh_num)))
        i = 0
        c = 0
        for k in wmh_num:
            if k == 0:
                c += 1
                continue
            # Weighted average penetration per slice
            wmh_cob_slice[c] = np.sum(wmh_cob[i:i + k] * wmh_area[i:i + k]) / max(np.sum(wmh_area[i:i + k]), 1)
            c += 1
            i += k

    # ==================== SLICE-WISE PENETRATION SCATTER PLOT ====================
    if len(wmh_cob_slice) == 0:        
        fig = go.Figure().update_layout(
            height=600, width=1100,
            xaxis_title='Slice Number',
            yaxis_title='Average Plaque Penetration per Slice (mm)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))

        # Save JSON data for empty slice-wise plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_penetration_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
                "slices_with_plaques": 0,
                "total_penetration_mm": 0.0
            },
            "data": {
                "slice_numbers": [],
                "penetration_per_slice": [],
                "plaques_per_slice": []
            }
        }

    else:
        x_values = np.arange(1, len(wmh_cob_slice) + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=wmh_cob_slice,
            mode='markers',
            marker=dict(
                size=8,
                color=color_codes['total'],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name='Penetration per Slice'
        ))
        
        fig.update_layout(
            height=600, width=1100,
            xaxis_title='Slice Number',
            yaxis_title='Weighted Average Plaque Penetration per Slice (mm)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            showlegend=False
        )
        
        fig.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Penetration (weighted average): {total_value} mm",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))
        
        # Save JSON data for slice-wise plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_penetration_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_cob_slice),
                "slices_with_plaques": int(np.sum(wmh_cob_slice > 0)),
                "total_penetration_mm": float(np.sum(wmh_cob_slice)),
                "mean_penetration_per_slice_mm": float(np.mean(wmh_cob_slice)),
                "median_penetration_per_slice_mm": float(np.median(wmh_cob_slice)),
                "std_penetration_per_slice_mm": float(np.std(wmh_cob_slice)),
                "max_penetration_per_slice_mm": float(np.max(wmh_cob_slice))
            },
            "data": {
                "slice_numbers": x_values.tolist(),
                "penetration_per_slice": (wmh_cob_slice).tolist(),
                "plaques_per_slice": wmh_num
            }
        }

    # Save figure and slice-wise JSON data
    pio.write_image(fig, os.path.join(save_path, f'Slices_Plaque_Penetration_tp{tp}.png'),
                    width=1100, height=600, scale=2)

    with open(os.path.join(save_path, f'slice_wise_penetration_scatter_data_tp{tp}.json'), 'w') as f:
        json.dump(scatter_slices_data, f, indent=2)

    # ==================== CALCULATE CATEGORY-SPECIFIC PENETRATION ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        peri_cd = []
        para_cd = []
        juxt_cd = []
        c_peri_s = []
        c_para_s = []
        c_juxt_s = []
        c_peri_ss = []
        c_para_ss = []
        c_juxt_ss = []
    else:
        # Calculate penetration by category
        peri_cd = np.where(np.array(wmh_code) == 1, 1, 0) * wmh_cob
        para_cd = np.where(np.array(wmh_code) == 2, 1, 0) * wmh_cob
        juxt_cd = np.where(np.array(wmh_code) == 3, 1, 0) * wmh_cob

        c_peri_s = np.zeros_like(np.array(wmh_cob_slice))
        c_peri_ss = np.zeros_like(np.array(wmh_cob_slice))
        c_para_s = np.zeros_like(np.array(wmh_cob_slice))
        c_para_ss = np.zeros_like(np.array(wmh_cob_slice))
        c_juxt_s = np.zeros_like(np.array(wmh_cob_slice))
        c_juxt_ss = np.zeros_like(np.array(wmh_cob_slice))
        
        i = 0
        c = 0
        whole_area = max(np.sum(wmh_area), 1)
        whole_area_peri = max(np.sum((np.where(np.array(wmh_code) == 1, 1, 0) * wmh_area)), 1)
        whole_area_para = max(np.sum((np.where(np.array(wmh_code) == 2, 1, 0) * wmh_area)), 1)
        whole_area_juxt = max(np.sum((np.where(np.array(wmh_code) == 3, 1, 0) * wmh_area)), 1)

        for k in wmh_num:
            if k == 0:
                c += 1
                continue
            # Calculate weighted penetration by category per slice
            c_peri_s[c] = np.sum(peri_cd[i:i + k] * wmh_area[i:i + k])
            c_peri_ss[c] = np.sum(peri_cd[i:i + k] * wmh_area[i:i + k]) / np.sum(np.where(peri_cd[i:i + k]>0, 1, 0) * wmh_area[i:i + k]) if np.sum(np.where(peri_cd[i:i + k]>0, 1, 0)) > 0 else 0
            c_para_s[c] = np.sum(para_cd[i:i + k] * wmh_area[i:i + k])
            c_para_ss[c] = np.sum(para_cd[i:i + k] * wmh_area[i:i + k]) / np.sum(np.where(para_cd[i:i + k]>0, 1, 0) * wmh_area[i:i + k]) if np.sum(np.where(para_cd[i:i + k]>0, 1, 0)) > 0 else 0
            c_juxt_s[c] = np.sum(juxt_cd[i:i + k] * wmh_area[i:i + k])
            c_juxt_ss[c] = np.sum(juxt_cd[i:i + k] * wmh_area[i:i + k]) / np.sum(np.where(juxt_cd[i:i + k]>0, 1, 0) * wmh_area[i:i + k]) if np.sum(np.where(juxt_cd[i:i + k]>0, 1, 0)) > 0 else 0
            c += 1
            i += k

    # ==================== GROUPED BAR PLOT FOR PENETRATION ====================
    nothing_to_show = False
    if len(wmh_num) == 0:
        nothing_to_show = True

    elif len(c_peri_ss) == 0 or len(c_para_ss) == 0 or len(c_juxt_ss) == 0:
        nothing_to_show = True

    if nothing_to_show:
        plot = go.Figure().update_layout(
            height=600, width=1200,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        plot.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for empty grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
                "periventricular_total": 0,
                "paraventricular_total": 0,
                "juxtacortical_total": 0
            },
            "data": {
                "slice_numbers": [],
                "periventricular_penetration": [],
                "paraventricular_penetration": [],
                "juxtacortical_penetration": []
            }
        }
        
    else:
        x = list(range(1, len(wmh_num) + 1))
        
        plot = go.Figure()
        
        # Create grouped bars with offset positions
        bar_width = 0.25
        
        # Note: Periventricular could be excluded as per original code logic
        plot.add_trace(go.Bar(
            name='Periventricular',
            x=[xi - bar_width for xi in x], 
            y=list(c_peri_ss),
            marker=dict(color=color_codes['peri']),
            width=bar_width,
            offsetgroup=1
        ))
        
        plot.add_trace(go.Bar(
            name='Paraventricular',
            x=x, 
            y=list(c_para_ss),
            marker=dict(color=color_codes['para']),
            width=bar_width,
            offsetgroup=2
        ))
        
        plot.add_trace(go.Bar(
            name='Juxtacortical',
            x=[xi + bar_width for xi in x], 
            y=list(c_juxt_ss),
            marker=dict(color=color_codes['juxt']),
            width=bar_width,
            offsetgroup=3
        ))
        
        plot.update_layout(
            barmode='group',
            height=600, width=1200,
            xaxis_title='Slice Number',
            yaxis_title='Weighted Average Plaque Penetration per Slice (mm)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5,
                tick0=1,
                dtick=1  # Ensure integer ticks for slice numbers
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            ),
            bargap=0.2,  # Gap between groups of bars
            bargroupgap=0.1  # Gap between bars within a group
        )
        
        # Save JSON data for grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_num),
                "periventricular_total": float(np.sum(c_peri_ss)),
                "paraventricular_total": float(np.sum(c_para_ss)),
                "juxtacortical_total": float(np.sum(c_juxt_ss)),
                "slices_with_peri": int(np.sum(c_peri_ss > 0)),
                "slices_with_para": int(np.sum(c_para_ss > 0)),
                "slices_with_juxt": int(np.sum(c_juxt_ss > 0))
            },
            "data": {
                "slice_numbers": x,
                "periventricular_penetration": (c_peri_ss).tolist(),
                "paraventricular_penetration": (c_para_ss).tolist(),
                "juxtacortical_penetration": (c_juxt_ss).tolist(),
                "bar_positions": {
                    "periventricular_x": [xi - bar_width for xi in x],
                    "paraventricular_x": x,
                    "juxtacortical_x": [xi + bar_width for xi in x]
                }
            }
        }

    # Save figure and JSON data
    pio.write_image(plot, os.path.join(save_path, f'Category_Plaque_Penetration_tp{tp}.png'),
                    width=1200, height=600, scale=2)

    with open(os.path.join(save_path, f'category_penetration_grouped_bar_data_tp{tp}.json'), 'w') as f:
        json.dump(grouped_bar_data, f, indent=2)

    # ==================== PIE CHART FOR PENETRATION ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0 or (np.sum(peri_cd) == 0 and np.sum(para_cd) == 0 and np.sum(juxt_cd) == 0):
        fig = go.Figure().update_layout(
            height=600, width=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No data to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for empty pie chart
        pie_chart_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "pie_chart_penetration_distribution",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_penetration_mm": 0.0,
                "categories_present": 0
            },
            "data": {
                "categories": [],
                "values": [],
                "percentages": [],
                "colors": []
            }
        }
        
    else:
        # Calculate totals for each category (excluding periventricular as in original)
        whole_area = max(np.sum(wmh_area), 1)
        peri_total = np.sum(c_peri_s) / whole_area if len(c_peri_s) > 0 else 0
        para_total = np.sum(c_para_s) / whole_area if len(c_para_s) > 0 else 0
        juxt_total = np.sum(c_juxt_s) / whole_area if len(c_juxt_s) > 0 else 0
        
        # Only include categories with non-zero values
        groups = []
        values = []
        colors = []
        
        if peri_total > 0:
            groups.append('Periventricular')
            values.append(peri_total)
            colors.append(color_codes['peri'])
            
        if para_total > 0:
            groups.append('Paraventricular')
            values.append(para_total)
            colors.append(color_codes['para'])
            
        if juxt_total > 0:
            groups.append('Juxtacortical')
            values.append(juxt_total)
            colors.append(color_codes['juxt'])
        
        if len(groups) == 0:
            fig = go.Figure().update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size)
            )
            fig.add_annotation(dict(
                font=dict(color='gray', size=18, family=font_family),
                x=0.5, y=0.5, showarrow=False,
                text="No data to display", textangle=0,
                xanchor='center', xref="paper", yref="paper"
            ))

            # Save JSON data for empty pie chart
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_penetration_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": False
                },
                "summary_statistics": {
                    "total_penetration_mm": 0,
                    "categories_present": 0
                },
                "data": {
                    "categories": [],
                    "values": [],
                    "percentages": [],
                    "colors": []
                }
            }

        else:
            total_penetration = sum(values)
            percentages = []
            
            # Create custom labels with penetration and percentage
            labels = []
            for i, (group, value, cat_area) in enumerate(zip(groups, values, [whole_area_peri, whole_area_para, whole_area_juxt])):
                percentage = (value / total_penetration) * 100
                percentages.append(percentage)
                value = (value * whole_area) / cat_area
                labels.append(f"{group}<br>{value:.1f} mm ({percentage:.1f}%)")
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                textfont=dict(size=14, family=font_family, color='black'),
                textinfo='label',
                textposition='inside',
                hole=0.3
            )])
            
            fig.update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size),
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50),
                title=dict(
                    text=f"Penetration Distribution (Total: {total_penetration:.1f} mm)",
                    x=0.5,
                    font=dict(size=title_font_size, family=font_family)
                )
            )
            
            # Save JSON data for populated pie chart
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_penetration_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": True
                },
                "summary_statistics": {
                    "total_penetration_mm": float(total_penetration),
                    "categories_present": len(groups)
                },
                "data": {
                    "categories": groups,
                    "values": [float(v) for v in values],
                    "percentages": [float(p) for p in percentages],
                    "colors": colors
                },
                "detailed_breakdown": {
                    "periventricular": {"penetration_mm": float(peri_total), "percentage": float((peri_total/total_penetration)*100) if total_penetration > 0 else 0.0},
                    "paraventricular": {"penetration_mm": float(para_total), "percentage": float((para_total/total_penetration)*100) if total_penetration > 0 else 0.0},
                    "juxtacortical": {"penetration_mm": float(juxt_total), "percentage": float((juxt_total/total_penetration)*100) if total_penetration > 0 else 0.0}
                }
            }
            
    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'Pie_Plaque_Penetration_tp{tp}.png'),
                    width=600, height=600, scale=2)

    with open(os.path.join(save_path, f'pie_chart_penetration_distribution_data_tp{tp}.json'), 'w') as f:
        json.dump(pie_chart_data, f, indent=2)

    # ==================== BOX-WHISKER PLOT FOR PENETRATION DISTRIBUTION ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0 or len(wmh_cob) == 0:
        fig = go.Figure().update_layout(
            height=600, width=800,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for empty box plot
        box_plot_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "box_plot_penetration_distribution",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_plaques": 0,
                "categories_with_data": []
            },
            "data": {
                "periventricular": {"penetration": [], "statistics": {}},
                "paraventricular": {"penetration": [], "statistics": {}},
                "juxtacortical": {"penetration": [], "statistics": {}}
            }
        }

    else:
        # Prepare data for box plots - separate penetration by category
        peri_cobs = wmh_cob[np.array(wmh_code) == 1] if np.any(np.array(wmh_code) == 1) else []
        para_cobs = wmh_cob[np.array(wmh_code) == 2] if np.any(np.array(wmh_code) == 2) else []
        juxt_cobs = wmh_cob[np.array(wmh_code) == 3] if np.any(np.array(wmh_code) == 3) else []
        
        fig = go.Figure()
        
        # Add box plots for each category (only if data exists)
        if len(peri_cobs) > 0:
            fig.add_trace(go.Box(
                y=peri_cobs,
                name='Periventricular',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['peri'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',  # Show outliers
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        if len(para_cobs) > 0:
            fig.add_trace(go.Box(
                y=para_cobs,
                name='Paraventricular',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['para'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        if len(juxt_cobs) > 0:
            fig.add_trace(go.Box(
                y=juxt_cobs,
                name='Juxtacortical',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['juxt'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        # If no data for any category, show empty message
        if len(peri_cobs) == 0 and len(para_cobs) == 0 and len(juxt_cobs) == 0:
            fig.add_annotation(dict(
                font=dict(color='gray', size=18, family=font_family),
                x=0.5, y=0.5, showarrow=False,
                text="No plaques to display", textangle=0,
                xanchor='center', xref="paper", yref="paper"
            ))
        
            # Save JSON data for box plot (no data case)
            box_plot_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "box_plot_penetration_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": False
                },
                "summary_statistics": {
                    "total_plaques": 0,
                    "categories_with_data": []
                },
                "data": {
                    "periventricular": {"penetration": [], "statistics": {}},
                    "paraventricular": {"penetration": [], "statistics": {}},
                    "juxtacortical": {"penetration": [], "statistics": {}}
                }
            }
        else:
            # Calculate detailed statistics for each category
            def calculate_box_stats(penetration):
                if len(penetration) == 0:
                    return {}
                penetration_array = np.array(penetration)
                return {
                    "count": len(penetration),
                    "mean": float(np.mean(penetration_array)),
                    "median": float(np.median(penetration_array)),
                    "std": float(np.std(penetration_array)),
                    "min": float(np.min(penetration_array)),
                    "max": float(np.max(penetration_array)),
                    "q1": float(np.percentile(penetration_array, 25)),
                    "q3": float(np.percentile(penetration_array, 75)),
                    "iqr": float(np.percentile(penetration_array, 75) - np.percentile(penetration_array, 25))
                }
            
            categories_with_data = []
            if len(peri_cobs) > 0: categories_with_data.append("periventricular")
            if len(para_cobs) > 0: categories_with_data.append("paraventricular")
            if len(juxt_cobs) > 0: categories_with_data.append("juxtacortical")
            
            # Save JSON data for box plot
            box_plot_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "box_plot_penetration_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": True
                },
                "summary_statistics": {
                    "total_plaques": len(wmh_cob),
                    "categories_with_data": categories_with_data,
                    "total_penetration_mm": float(total_value)
                },
                "data": {
                    "periventricular": {
                        "penetration": peri_cobs.tolist() if len(peri_cobs) > 0 else [],
                        "statistics": calculate_box_stats(peri_cobs)
                    },
                    "paraventricular": {
                        "penetration": para_cobs.tolist() if len(para_cobs) > 0 else [],
                        "statistics": calculate_box_stats(para_cobs)
                    },
                    "juxtacortical": {
                        "penetration": juxt_cobs.tolist() if len(juxt_cobs) > 0 else [],
                        "statistics": calculate_box_stats(juxt_cobs)
                    }
                }
            }

        fig.update_layout(
            height=600, width=800,
            xaxis_title='Plaque Categories',
            yaxis_title='Plaque Penetration (mm)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        # Add summary statistics annotation
        total_plaques = len(wmh_cob)
        fig.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size-2, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Plaques: {total_plaques} | Total Penetration: {total_value} mm",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'BoxPlot_Plaque_Penetration_tp{tp}.png'),
                    width=600, height=600, scale=2)

    with open(os.path.join(save_path, f'box_plot_penetration_distribution_data_tp{tp}.json'), 'w') as f:
        json.dump(box_plot_data, f, indent=2)

    # ==================== DATA PROCESSING FOR OUTPUT ====================
    if len(c_peri_s) == 0 or len(c_para_s) == 0 or len(c_juxt_s) == 0:
        c_peri_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        c_para_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        c_juxt_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
    else:
        whole_area = max(np.sum(wmh_area), 1)
        
        c_peri_ss = np.zeros_like(c_peri_s)
        c_para_ss = np.zeros_like(c_peri_s)
        c_juxt_ss = np.zeros_like(c_peri_s)

        c_peri_ss[0:len(c_peri_s)] = c_peri_s / whole_area
        c_para_ss[0:len(c_para_s)] = c_para_s / whole_area
        c_juxt_ss[0:len(c_juxt_s)] = c_juxt_s / whole_area

    # ==================== SAVE COMPREHENSIVE SUMMARY JSON ====================
    # Create a comprehensive summary file with all key data
    comprehensive_summary = {
        "metadata": {
            "subject_id": id,
            "timepoint": tp,
            "analysis_type": "wmh_penetration_analysis",
            "timestamp": datetime.now().isoformat(),
            "generated_plots": [
                f"All_Plaque_Penetration_tp{tp}.png",
                f"Slices_Plaque_Penetration_tp{tp}.png", 
                f"Category_Plaque_Penetration_tp{tp}.png",
                f"Pie_Plaque_Penetration_tp{tp}.png",
                f"BoxPlot_Plaque_Penetration_tp{tp}.png"
            ],
            "generated_data_files": [
                f"sorted_penetration_scatter_data_tp{tp}.json",
                f"slice_wise_penetration_scatter_data_tp{tp}.json",
                f"category_penetration_grouped_bar_data_tp{tp}.json", 
                f"pie_chart_penetration_distribution_data_tp{tp}.json",
                f"box_plot_penetration_distribution_data_tp{tp}.json",
                f"comprehensive_penetration_summary_tp{tp}.json"
            ]
        },
        "global_statistics": {
            "total_plaques": len(wmh_cob) if len(wmh_cob) > 0 else 0,
            "total_penetration_mm": float(total_value) if not np.isnan(total_value) and not np.isinf(total_value) else 0.0,
            "penetration_statistics": {
                "mean_mm": float(np.mean(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.mean(wmh_cob)) else 0.0,
                "median_mm": float(np.median(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.median(wmh_cob)) else 0.0,
                "std_mm": float(np.std(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.std(wmh_cob)) else 0.0,
                "min_mm": float(np.min(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.min(wmh_cob)) else 0.0,
                "max_mm": float(np.max(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.max(wmh_cob)) else 0.0
            }
        },
        "category_breakdown": {
            "periventricular": {
                "count": int(np.sum(np.array(wmh_code) == 1)) if len(wmh_code) > 0 else 0,
                "total_penetration_mm": float(np.sum(c_peri_ss)) if len(c_peri_ss) > 0 and not np.isnan(np.sum(c_peri_ss)) else 0.0,
                "percentage_of_total": float((np.sum(c_peri_ss) / total_value * 100)) if total_value > 0 and not np.isnan(total_value) and not np.isinf(total_value) else 0.0
            },
            "paraventricular": {
                "count": int(np.sum(np.array(wmh_code) == 2)) if len(wmh_code) > 0 else 0,
                "total_penetration_mm": float(np.sum(c_para_ss)) if len(c_para_ss) > 0 and not np.isnan(np.sum(c_para_ss)) else 0.0,
                "percentage_of_total": float((np.sum(c_para_ss) / total_value * 100)) if total_value > 0 and not np.isnan(total_value) and not np.isinf(total_value) else 0.0
            },
            "juxtacortical": {
                "count": int(np.sum(np.array(wmh_code) == 3)) if len(wmh_code) > 0 else 0,
                "total_penetration_mm": float(np.sum(c_juxt_ss)) if len(c_juxt_ss) > 0 and not np.isnan(np.sum(c_juxt_ss)) else 0.0,
                "percentage_of_total": float((np.sum(c_juxt_ss) / total_value * 100)) if total_value > 0 and not np.isnan(total_value) and not np.isinf(total_value) else 0.0
            }
        },
        "slice_analysis": {
            "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
            "slices_with_plaques": int(np.sum(np.array(wmh_cob_slice) > 0)) if len(wmh_cob_slice) > 0 else 0,
            "slice_penetration_statistics": {
                "mean_penetration_per_slice_mm": float(np.mean(wmh_cob_slice)) if len(wmh_cob_slice) > 0 and not np.isnan(np.mean(wmh_cob_slice)) else 0.0,
                "max_penetration_per_slice_mm": float(np.max(wmh_cob_slice)) if len(wmh_cob_slice) > 0 and not np.isnan(np.max(wmh_cob_slice)) else 0.0,
                "std_penetration_per_slice_mm": float(np.std(wmh_cob_slice)) if len(wmh_cob_slice) > 0 and not np.isnan(np.std(wmh_cob_slice)) else 0.0
            }
        },
        "raw_data": {
            "wmh_areas": wmh_area.tolist() if hasattr(wmh_area, 'tolist') and len(wmh_area) > 0 else (list(wmh_area) if len(wmh_area) > 0 else []),
            "wmh_cobs": wmh_cob.tolist() if hasattr(wmh_cob, 'tolist') and len(wmh_cob) > 0 else (list(wmh_cob) if len(wmh_cob) > 0 else []),
            "wmh_codes": list(wmh_code) if len(wmh_code) > 0 else [],
            "wmh_num_per_slice": list(wmh_num) if len(wmh_num) > 0 else [],
            "penetration_per_slice": wmh_cob_slice.tolist() if hasattr(wmh_cob_slice, 'tolist') and len(wmh_cob_slice) > 0 else (list(wmh_cob_slice) if len(wmh_cob_slice) > 0 else []),
            "category_penetration_per_slice": {
                "periventricular": c_peri_ss.tolist() if hasattr(c_peri_ss, 'tolist') else list(c_peri_ss),
                "paraventricular": c_para_ss.tolist() if hasattr(c_para_ss, 'tolist') else list(c_para_ss),
                "juxtacortical": c_juxt_ss.tolist() if hasattr(c_juxt_ss, 'tolist') else list(c_juxt_ss)
            }
        }
    }

    # Convert the entire dictionary to ensure all NumPy types are handled
    comprehensive_summary = convert_numpy_to_serializable(comprehensive_summary)

    # Save comprehensive summary with error handling
    try:
        with open(os.path.join(save_path, f'comprehensive_penetration_summary_tp{tp}.json'), 'w') as f:
            json.dump(comprehensive_summary, f, indent=2)
        print(f"Successfully saved comprehensive summary to comprehensive_penetration_summary_tp{tp}.json")
    except Exception as e:
        print(f"Error saving comprehensive summary: {e}")
        # Optionally save a simplified version or debug info
        print("Attempting to save simplified version...")
        try:
            # Create a simplified version with just the basic info
            simplified_summary = {
                "metadata": comprehensive_summary["metadata"],
                "global_statistics": {
                    "total_plaques": comprehensive_summary["global_statistics"]["total_plaques"],
                    "total_penetration_mm": comprehensive_summary["global_statistics"]["total_penetration_mm"]
                },
                "error_info": {
                    "original_error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
            with open(os.path.join(save_path, f'simplified_penetration_summary_tp{tp}.json'), 'w') as f:
                json.dump(simplified_summary, f, indent=2)
            print("Successfully saved simplified summary")
        except Exception as e2:
            print(f"Failed to save even simplified summary: {e2}")

            
    # # Set periventricular to zero as per original logic
    # c_peri_ss[:] = 0

    # Update metadata
    penetration_general_data = {
        "penetration_of_all_found_WMH": list(np.uint16(wmh_cob)) if len(wmh_cob) > 0 else [],
        "whole_brain_total_penetration": int(total_value)
    }
    penetration_periventricular_data = {
        "whole_brain_total_penetration": int(np.sum(c_peri_ss))
    }
    penetration_paraventricular_data = {
        "whole_brain_total_penetration": int(np.sum(c_para_ss))
    }
    penetration_juxtacortical_data = {
        "whole_brain_total_penetration": int(np.sum(c_juxt_ss))
    }

    # Note: These function calls will need to be uncommented if the functions exist
    update_wmh_data(wmh_m_data, "penetration", "general", penetration_general_data)
    update_wmh_data(wmh_m_data, "penetration", "periventricular", penetration_periventricular_data)
    update_wmh_data(wmh_m_data, "penetration", "paraventricular", penetration_paraventricular_data)
    update_wmh_data(wmh_m_data, "penetration", "juxtacortical", penetration_juxtacortical_data)

    return [c_peri_ss, c_para_ss, c_juxt_ss]

#
def Ddist_show_subj(save_path, wmh_cob, wmh_area, wmh_code, wmh_num, id, tp=0):
    """
    Enhanced function for publication-ready WMH lesion depth visualization
    Consistent with the area_show_subj function styling and structure
    Now includes JSON data export for each visualization
    """

    if len(wmh_cob) == 0:
        total_value = int(0)
        print(f"There is no seen plaque.")
    else:
        wmh_cob = np.array(wmh_cob)
        wmh_area = np.round(np.array(wmh_area), 1)
        wmh_code = np.array(wmh_code)
        
        # Calculate weighted average Depth
        total_value = int(np.round(np.sum(wmh_cob * wmh_area) / max(np.sum(wmh_area), 1)))
        print(f"Total Depth of Plaques (weighted average): {total_value} mm")

    # ==================== SORTED Depth SCATTER PLOT WITH CATEGORY COLORS ====================
    if len(wmh_cob) == 0:
        fig = go.Figure().update_layout(
            height=600, width=1100,
            xaxis_title='Plaque Index (Sorted by Depth)',
            yaxis_title='Plaque Depth (mm)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for empty sorted depth plot
        scatter_sorted_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "sorted_depths_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_plaques": 0,
                "total_depths_mm": 0.0,
            },
            "data": {
                "plaque_depths": [],
                "plaque_codes": [],
                "sorted_indices": [],
                "x_values": [],
                "y_values": []
            },
            "categories": {
                "periventricular": {"count": 0, "depths": []},
                "paraventricular": {"count": 0, "depths": []},
                "juxtacortical": {"count": 0, "depths": []}
            }
        }
    else:
        # Sort Depth values while keeping track of original indices
        sorted_indices = np.argsort(wmh_cob)
        sorted_penetration = wmh_cob[sorted_indices]
        sorted_codes = np.array(wmh_code)[sorted_indices]
        x_values = np.arange(1, len(wmh_cob) + 1)
    
        fig = go.Figure()
        
        # Create separate traces for each category to enable proper legend
        categories = {
            1: {'name': 'Periventricular', 'color': color_codes['peri']},
            2: {'name': 'Paraventricular', 'color': color_codes['para']},
            3: {'name': 'Juxtacortical', 'color': color_codes['juxt']}
        }

        # Add traces for each category
        for cat_code, cat_key in categories.items():
            mask = sorted_codes == cat_code
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=x_values[mask], 
                    y=sorted_penetration[mask],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cat_key['color'],
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    name=cat_key['name'],
                    legendgroup=cat_key['name']
                ))
        
        fig.update_layout(
            height=600, width=1100,
            xaxis_title='Plaque Index (Sorted by Depth)',
            yaxis_title='Plaque Depth (mm)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            )
        )
        
        # fig.add_annotation(dict(
        #     font=dict(color=color_codes['total'], size=annotation_font_size, family=font_family),
        #     x=0.98, y=0.95, showarrow=False,
        #     text=f"Total Depth (weighted average): {total_value} mm",
        #     textangle=0, xanchor='right', xref="paper", yref="paper",
        #     bgcolor="rgba(255,255,255,0.8)",
        #     bordercolor=color_codes['total'],
        #     borderwidth=1
        # ))

        # Save JSON data for sorted scatter plot
        scatter_sorted_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "sorted_depths_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_plaques": len(wmh_cob),
                "total_depth_mm": float(total_value),
                "mean_depth_mm": float(np.mean(wmh_cob)),
                "median_depth_mm": float(np.median(wmh_cob)),
                "std_depth_mm": float(np.std(wmh_cob)),
                "min_depth_mm": float(np.min(wmh_cob)),
                "max_depth_mm": float(np.max(wmh_cob))
            },
            "data": {
                "original_depth": wmh_cob.tolist(),
                "original_codes": np.array(wmh_code).tolist(),
                "sorted_indices": sorted_indices.tolist(),
                "sorted_depth": sorted_penetration.tolist(),
                "sorted_codes": sorted_codes.tolist(),
                "x_values": x_values.tolist(),
                "y_values": sorted_penetration.tolist()
            },
            "categories": {
                "periventricular": {
                    "count": int(np.sum(np.array(wmh_code) == 1)),
                    "depth": wmh_cob[np.array(wmh_code) == 1].tolist(),
                    "total_depth_mm": float(np.sum(wmh_cob[np.array(wmh_code) == 1]))
                },
                "paraventricular": {
                    "count": int(np.sum(np.array(wmh_code) == 2)),
                    "depth": wmh_cob[np.array(wmh_code) == 2].tolist(),
                    "total_depth_mm": float(np.sum(wmh_cob[np.array(wmh_code) == 2]))
                },
                "juxtacortical": {
                    "count": int(np.sum(np.array(wmh_code) == 3)),
                    "depth": wmh_cob[np.array(wmh_code) == 3].tolist(),
                    "total_depth_mm": float(np.sum(wmh_cob[np.array(wmh_code) == 3]))
                }
            }
        }

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'All_Plaque_Depth_tp{tp}.png'),
                    width=1200, height=600, scale=2)

    with open(os.path.join(save_path, f'sorted_depth_scatter_data_tp{tp}.json'), 'w') as f:
        json.dump(scatter_sorted_data, f, indent=2)

    # ==================== CALCULATE DEPTH PER SLICE ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        wmh_cob_slice = []
    else:
        wmh_cob_slice = np.zeros((len(wmh_num)))
        i = 0
        c = 0
        for k in wmh_num:
            if k == 0:
                c += 1
                continue
            # Weighted average Depth per slice
            wmh_cob_slice[c] = np.sum(wmh_cob[i:i + k] * wmh_area[i:i + k]) / max(np.sum(wmh_area[i:i + k]), 1)
            c += 1
            i += k

    # ==================== SLICE-WISE DEPTH SCATTER PLOT ====================
    if len(wmh_cob_slice) == 0:
        fig = go.Figure().update_layout(
            height=600, width=1100,
            xaxis_title='Slice Number',
            yaxis_title='Average Plaque Depth per Slice (mm)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        
        # Save JSON data for empty slice-wise plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_depth_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
                "slices_with_plaques": 0,
                "total_depth_mm": 0.0
            },
            "data": {
                "slice_numbers": [],
                "depth_per_slice": [],
                "plaques_per_slice": []
            }
        }

    else:
        x_values = np.arange(1, len(wmh_cob_slice) + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=wmh_cob_slice,
            mode='markers',
            marker=dict(
                size=8,
                color=color_codes['total'],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name='Depth per Slice'
        ))
        
        fig.update_layout(
            height=600, width=1100,
            xaxis_title='Slice Number',
            yaxis_title='Weighted Average Plaque Depth per Slice (mm)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            showlegend=False
        )
        
        fig.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Depth (weighted average): {total_value} mm",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))
        
        # Save JSON data for slice-wise plot
        scatter_slices_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "slice_wise_depth_scatter",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_cob_slice),
                "slices_with_plaques": int(np.sum(wmh_cob_slice > 0)),
                "total_depth_mm": float(np.sum(wmh_cob_slice)),
                "mean_depth_per_slice_mm": float(np.mean(wmh_cob_slice)),
                "median_depth_per_slice_mm": float(np.median(wmh_cob_slice)),
                "std_depth_per_slice_mm": float(np.std(wmh_cob_slice)),
                "max_depth_per_slice_mm": float(np.max(wmh_cob_slice))
            },
            "data": {
                "slice_numbers": x_values.tolist(),
                "depth_per_slice": (wmh_cob_slice).tolist(),
                "plaques_per_slice": wmh_num
            }
        }

    # Save figure and slice-wise JSON data
    pio.write_image(fig, os.path.join(save_path, f'Slices_Plaque_Depth_tp{tp}.png'),
                    width=1100, height=600, scale=2)

    with open(os.path.join(save_path, f'slice_wise_depth_scatter_data_tp{tp}.json'), 'w') as f:
        json.dump(scatter_slices_data, f, indent=2)

    # ==================== CALCULATE CATEGORY-SPECIFIC DEPTH ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0:
        peri_cd = []
        para_cd = []
        juxt_cd = []
        c_peri_s = []
        c_para_s = []
        c_juxt_s = []
        c_peri_ss = []
        c_para_ss = []
        c_juxt_ss = []
    else:
        # Calculate penetration by category
        peri_cd = np.where(np.array(wmh_code) == 1, 1, 0) * wmh_cob
        para_cd = np.where(np.array(wmh_code) == 2, 1, 0) * wmh_cob
        juxt_cd = np.where(np.array(wmh_code) == 3, 1, 0) * wmh_cob

        c_peri_s = np.zeros_like(np.array(wmh_cob_slice))
        c_peri_ss = np.zeros_like(np.array(wmh_cob_slice))
        c_para_s = np.zeros_like(np.array(wmh_cob_slice))
        c_para_ss = np.zeros_like(np.array(wmh_cob_slice))
        c_juxt_s = np.zeros_like(np.array(wmh_cob_slice))
        c_juxt_ss = np.zeros_like(np.array(wmh_cob_slice))
        
        i = 0
        c = 0
        whole_area = max(np.sum(wmh_area), 1)
        whole_area_peri = max(np.sum((np.where(np.array(wmh_code) == 1, 1, 0) * wmh_area)), 1)
        whole_area_para = max(np.sum((np.where(np.array(wmh_code) == 2, 1, 0) * wmh_area)), 1)
        whole_area_juxt = max(np.sum((np.where(np.array(wmh_code) == 3, 1, 0) * wmh_area)), 1)

        for k in wmh_num:
            if k == 0:
                c += 1
                continue
            # Calculate weighted penetration by category per slice
            c_peri_s[c] = np.sum(peri_cd[i:i + k] * wmh_area[i:i + k])
            c_peri_ss[c] = np.sum(peri_cd[i:i + k] * wmh_area[i:i + k]) / np.sum(np.where(peri_cd[i:i + k]>0, 1, 0) * wmh_area[i:i + k]) if np.sum(np.where(peri_cd[i:i + k]>0, 1, 0)) > 0 else 0
            c_para_s[c] = np.sum(para_cd[i:i + k] * wmh_area[i:i + k])
            c_para_ss[c] = np.sum(para_cd[i:i + k] * wmh_area[i:i + k]) / np.sum(np.where(para_cd[i:i + k]>0, 1, 0) * wmh_area[i:i + k]) if np.sum(np.where(para_cd[i:i + k]>0, 1, 0)) > 0 else 0
            c_juxt_s[c] = np.sum(juxt_cd[i:i + k] * wmh_area[i:i + k])
            c_juxt_ss[c] = np.sum(juxt_cd[i:i + k] * wmh_area[i:i + k]) / np.sum(np.where(juxt_cd[i:i + k]>0, 1, 0) * wmh_area[i:i + k]) if np.sum(np.where(juxt_cd[i:i + k]>0, 1, 0)) > 0 else 0
            c += 1
            i += k

    # ==================== GROUPED BAR PLOT FOR PENETRATION ====================
    nothing_to_show = False
    if len(wmh_num) == 0:
        nothing_to_show = True

    elif len(c_peri_ss) == 0 or len(c_para_ss) == 0 or len(c_juxt_ss) == 0:
        nothing_to_show = True

    if nothing_to_show:
        plot = go.Figure().update_layout(
            height=600, width=1200,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        plot.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for empty grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
                "periventricular_total": 0,
                "paraventricular_total": 0,
                "juxtacortical_total": 0
            },
            "data": {
                "slice_numbers": [],
                "periventricular_depth": [],
                "paraventricular_depth": [],
                "juxtacortical_depth": []
            }
        }

    else:
        x = list(range(1, len(wmh_num) + 1))
        
        plot = go.Figure()
        
        # Create grouped bars with offset positions
        bar_width = 0.25
        
        # Note: Periventricular could be excluded as per original code logic
        plot.add_trace(go.Bar(
            name='Periventricular',
            x=[xi - bar_width for xi in x], 
            y=list(c_peri_ss),
            marker=dict(color=color_codes['peri']),
            width=bar_width,
            offsetgroup=1
        ))
        
        plot.add_trace(go.Bar(
            name='Paraventricular',
            x=x, 
            y=list(c_para_ss),
            marker=dict(color=color_codes['para']),
            width=bar_width,
            offsetgroup=2
        ))
        
        plot.add_trace(go.Bar(
            name='Juxtacortical',
            x=[xi + bar_width for xi in x], 
            y=list(c_juxt_ss),
            marker=dict(color=color_codes['juxt']),
            width=bar_width,
            offsetgroup=3
        ))
        
        plot.update_layout(
            barmode='group',
            height=600, width=1200,
            xaxis_title='Slice Number',
            yaxis_title='Weighted Average Plaque Depth per Slice (mm)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5,
                tick0=1,
                dtick=1  # Ensure integer ticks for slice numbers
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=legend_font_size, family=font_family)
            ),
            bargap=0.2,  # Gap between groups of bars
            bargroupgap=0.1  # Gap between bars within a group
        )

        # # Add category-specific annotations
        # whole_area = max(np.sum(wmh_area), 1)
        # plot.add_annotation(dict(
        #     font=dict(color=color_codes['para'], size=annotation_font_size, family=font_family),
        #     x=0.98, y=0.95, showarrow=False,
        #     text=f"Paraventricular: {int(np.round(np.sum(c_para_s) / whole_area))} mm",
        #     textangle=0, xanchor='right', xref="paper", yref="paper",
        #     bgcolor="rgba(255,255,255,0.8)",
        #     bordercolor=color_codes['para'],
        #     borderwidth=1
        # ))
        # plot.add_annotation(dict(
        #     font=dict(color=color_codes['juxt'], size=annotation_font_size, family=font_family),
        #     x=0.98, y=0.88, showarrow=False,
        #     text=f"Juxtacortical: {int(np.round(np.sum(c_juxt_s) / whole_area))} mm",
        #     textangle=0, xanchor='right', xref="paper", yref="paper",
        #     bgcolor="rgba(255,255,255,0.8)",
        #     bordercolor=color_codes['juxt'],
        #     borderwidth=1
        # ))

        # Save JSON data for grouped bar plot
        grouped_bar_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "grouped_bar_categories",
                "timestamp": datetime.now().isoformat(),
                "has_data": True
            },
            "summary_statistics": {
                "total_slices": len(wmh_num),
                "periventricular_total": float(np.sum(c_peri_ss)),
                "paraventricular_total": float(np.sum(c_para_ss)),
                "juxtacortical_total": float(np.sum(c_juxt_ss)),
                "slices_with_peri": int(np.sum(c_peri_ss > 0)),
                "slices_with_para": int(np.sum(c_para_ss > 0)),
                "slices_with_juxt": int(np.sum(c_juxt_ss > 0))
            },
            "data": {
                "slice_numbers": x,
                "periventricular_depth": (c_peri_ss).tolist(),
                "paraventricular_depth": (c_para_ss).tolist(),
                "juxtacortical_depth": (c_juxt_ss).tolist(),
                "bar_positions": {
                    "periventricular_x": [xi - bar_width for xi in x],
                    "paraventricular_x": x,
                    "juxtacortical_x": [xi + bar_width for xi in x]
                }
            }
        }

    # Save figure and JSON data
    pio.write_image(plot, os.path.join(save_path, f'Category_Plaque_Depth_tp{tp}.png'),
                    width=1200, height=600, scale=2)

    with open(os.path.join(save_path, f'category_depth_grouped_bar_data_tp{tp}.json'), 'w') as f:
        json.dump(grouped_bar_data, f, indent=2)

    # ==================== PIE CHART FOR Depth ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0 or (np.sum(peri_cd) == 0 and np.sum(para_cd) == 0 and np.sum(juxt_cd) == 0):
        fig = go.Figure().update_layout(
            height=600, width=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No data to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for empty pie chart
        pie_chart_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "pie_chart_depth_distribution",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_depth_mm": 0.0,
                "categories_present": 0
            },
            "data": {
                "categories": [],
                "values": [],
                "percentages": [],
                "colors": []
            }
        }

    else:
        # Calculate totals for each category (maybe excluding periventricular as in original)
        whole_area = max(np.sum(wmh_area), 1)
        peri_total = np.sum(c_peri_s) / whole_area if len(c_peri_s) > 0 else 0
        para_total = np.sum(c_para_s) / whole_area if len(c_para_s) > 0 else 0
        juxt_total = np.sum(c_juxt_s) / whole_area if len(c_juxt_s) > 0 else 0
        
        # Only include categories with non-zero values
        groups = []
        values = []
        colors = []
        
        if peri_total > 0:
            groups.append('Periventricular')
            values.append(peri_total)
            colors.append(color_codes['peri'])
        
        if para_total > 0:
            groups.append('Paraventricular')
            values.append(para_total)
            colors.append(color_codes['para'])
            
        if juxt_total > 0:
            groups.append('Juxtacortical')
            values.append(juxt_total)
            colors.append(color_codes['juxt'])
        
        if len(groups) == 0:
            fig = go.Figure().update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size)
            )
            fig.add_annotation(dict(
                font=dict(color='gray', size=18, family=font_family),
                x=0.5, y=0.5, showarrow=False,
                text="No data to display", textangle=0,
                xanchor='center', xref="paper", yref="paper"
            ))

            # Save JSON data for empty pie chart
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_depth_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": False
                },
                "summary_statistics": {
                    "total_depth_mm": 0,
                    "categories_present": 0
                },
                "data": {
                    "categories": [],
                    "values": [],
                    "percentages": [],
                    "colors": []
                }
            }

        else:
            total_depth = sum(values)
            percentages = []
            
            # Create custom labels with penetration and percentage
            labels = []
            for i, (group, value, cat_area) in enumerate(zip(groups, values, [whole_area_peri, whole_area_para, whole_area_juxt])):
                percentage = (value / total_depth) * 100
                percentages.append(percentage)
                value = (value * whole_area) / cat_area
                labels.append(f"{group}<br>{value:.1f} mm ({percentage:.1f}%)")
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                textfont=dict(size=14, family=font_family, color='black'),
                textinfo='label',
                textposition='inside',
                hole=0.3
            )])
            
            fig.update_layout(
                height=600, width=600,
                font=dict(family=font_family, size=axis_font_size),
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50),
                title=dict(
                    text=f"Depth Distribution (Total: {total_depth:.1f} mm)",
                    x=0.5,
                    font=dict(size=title_font_size, family=font_family)
                )
            )
            
            # Save JSON data for populated pie chart
            pie_chart_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "pie_chart_depth_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": True
                },
                "summary_statistics": {
                    "total_depth_mm": float(total_depth),
                    "categories_present": len(groups)
                },
                "data": {
                    "categories": groups,
                    "values": [float(v) for v in values],
                    "percentages": [float(p) for p in percentages],
                    "colors": colors
                },
                "detailed_breakdown": {
                    "periventricular": {"depth_mm": float(peri_total), "percentage": float((peri_total/total_depth)*100) if total_depth > 0 else 0.0},
                    "paraventricular": {"depth_mm": float(para_total), "percentage": float((para_total/total_depth)*100) if total_depth > 0 else 0.0},
                    "juxtacortical": {"depth_mm": float(juxt_total), "percentage": float((juxt_total/total_depth)*100) if total_depth > 0 else 0.0}
                }
            }
            
    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'Pie_Plaque_Depth_tp{tp}.png'),
                    width=600, height=600, scale=2)

    with open(os.path.join(save_path, f'pie_chart_depth_distribution_data_tp{tp}.json'), 'w') as f:
        json.dump(pie_chart_data, f, indent=2)

    # ==================== BOX-WHISKER PLOT FOR LOCATION DISTRIBUTION ====================
    if len(wmh_num) == 0 or np.sum(wmh_num) == 0 or len(wmh_cob) == 0:
        fig = go.Figure().update_layout(
            height=600, width=800,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            font=dict(family=font_family, size=axis_font_size)
        )
        fig.add_annotation(dict(
            font=dict(color='gray', size=18, family=font_family),
            x=0.5, y=0.5, showarrow=False,
            text="No plaques to display", textangle=0,
            xanchor='center', xref="paper", yref="paper"
        ))
        # Save JSON data for empty box plot
        box_plot_data = {
            "metadata": {
                "subject_id": id,
                "timepoint": tp,
                "plot_type": "box_plot_depth_distribution",
                "timestamp": datetime.now().isoformat(),
                "has_data": False
            },
            "summary_statistics": {
                "total_plaques": 0,
                "categories_with_data": []
            },
            "data": {
                "periventricular": {"depth": [], "statistics": {}},
                "paraventricular": {"depth": [], "statistics": {}},
                "juxtacortical": {"depth": [], "statistics": {}}
            }
        }

    else:
        # Prepare data for box plots - separate penetration by category
        peri_cobs = wmh_cob[np.array(wmh_code) == 1] if np.any(np.array(wmh_code) == 1) else []
        para_cobs = wmh_cob[np.array(wmh_code) == 2] if np.any(np.array(wmh_code) == 2) else []
        juxt_cobs = wmh_cob[np.array(wmh_code) == 3] if np.any(np.array(wmh_code) == 3) else []
        
        fig = go.Figure()
        
        # Add box plots for each category (only if data exists)
        if len(peri_cobs) > 0:
            fig.add_trace(go.Box(
                y=peri_cobs,
                name='Periventricular',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['peri'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',  # Show outliers
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        if len(para_cobs) > 0:
            fig.add_trace(go.Box(
                y=para_cobs,
                name='Paraventricular',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['para'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        if len(juxt_cobs) > 0:
            fig.add_trace(go.Box(
                y=juxt_cobs,
                name='Juxtacortical',
                marker=dict(color='black'),  # Outlier points in black
                fillcolor=color_codes['juxt'],  # Inside fill with category color
                line=dict(color='black', width=1.5),  # Box outline in black
                boxpoints=False, #'outliers',
                jitter=0.3,
                pointpos=-1.8,
                boxmean=False  # Shows mean as a line inside the box
            ))
        
        # If no data for any category, show empty message
        if len(peri_cobs) == 0 and len(para_cobs) == 0 and len(juxt_cobs) == 0:
            fig.add_annotation(dict(
                font=dict(color='gray', size=18, family=font_family),
                x=0.5, y=0.5, showarrow=False,
                text="No plaques to display", textangle=0,
                xanchor='center', xref="paper", yref="paper"
            ))

            # Save JSON data for box plot (no data case)
            box_plot_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "box_plot_depth_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": False
                },
                "summary_statistics": {
                    "total_plaques": 0,
                    "categories_with_data": []
                },
                "data": {
                    "periventricular": {"depth": [], "statistics": {}},
                    "paraventricular": {"depth": [], "statistics": {}},
                    "juxtacortical": {"depth": [], "statistics": {}}
                }
            }
        else:
            # Calculate detailed statistics for each category
            def calculate_box_stats(depth):
                if len(depth) == 0:
                    return {}
                depth_array = np.array(depth)
                return {
                    "count": len(depth),
                    "mean": float(np.mean(depth_array)),
                    "median": float(np.median(depth_array)),
                    "std": float(np.std(depth_array)),
                    "min": float(np.min(depth_array)),
                    "max": float(np.max(depth_array)),
                    "q1": float(np.percentile(depth_array, 25)),
                    "q3": float(np.percentile(depth_array, 75)),
                    "iqr": float(np.percentile(depth_array, 75) - np.percentile(depth_array, 25))
                }
            
            categories_with_data = []
            if len(peri_cobs) > 0: categories_with_data.append("periventricular")
            if len(para_cobs) > 0: categories_with_data.append("paraventricular")
            if len(juxt_cobs) > 0: categories_with_data.append("juxtacortical")
            
            # Save JSON data for box plot
            box_plot_data = {
                "metadata": {
                    "subject_id": id,
                    "timepoint": tp,
                    "plot_type": "box_plot_depth_distribution",
                    "timestamp": datetime.now().isoformat(),
                    "has_data": True
                },
                "summary_statistics": {
                    "total_plaques": len(wmh_cob),
                    "categories_with_data": categories_with_data,
                    "total_depth_mm": float(total_value)
                },
                "data": {
                    "periventricular": {
                        "depth": peri_cobs.tolist() if len(peri_cobs) > 0 else [],
                        "statistics": calculate_box_stats(peri_cobs)
                    },
                    "paraventricular": {
                        "depth": para_cobs.tolist() if len(para_cobs) > 0 else [],
                        "statistics": calculate_box_stats(para_cobs)
                    },
                    "juxtacortical": {
                        "depth": juxt_cobs.tolist() if len(juxt_cobs) > 0 else [],
                        "statistics": calculate_box_stats(juxt_cobs)
                    }
                }
            }
            
        fig.update_layout(
            height=600, width=800,
            xaxis_title='Plaque Categories',
            yaxis_title='Plaque Depth (mm)',
            font=dict(family=font_family, size=axis_font_size),
            xaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            yaxis=dict(
                titlefont=dict(size=title_font_size, family=font_family),
                tickfont=dict(size=axis_font_size, family=font_family),
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            showlegend=False,
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        # Add summary statistics annotation
        total_plaques = len(wmh_cob)
        fig.add_annotation(dict(
            font=dict(color=color_codes['total'], size=annotation_font_size-2, family=font_family),
            x=0.02, y=0.95, showarrow=False,
            text=f"Total Plaques: {total_plaques} | Total Depth: {total_value} mm",
            textangle=0, xanchor='left', xref="paper", yref="paper",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_codes['total'],
            borderwidth=1
        ))

    # Save figure and JSON data
    pio.write_image(fig, os.path.join(save_path, f'BoxPlot_Plaque_Depth_tp{tp}.png'),
                    width=600, height=600, scale=2)

    with open(os.path.join(save_path, f'box_plot_depth_distribution_data_tp{tp}.json'), 'w') as f:
        json.dump(box_plot_data, f, indent=2)

    # ==================== DATA PROCESSING FOR OUTPUT ====================
    if len(c_peri_s) == 0 or len(c_para_s) == 0 or len(c_juxt_s) == 0:
        c_peri_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        c_para_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
        c_juxt_ss = np.zeros((len(wmh_num))) if len(wmh_num) > 0 else np.zeros(1)
    else:
        whole_area = max(np.sum(wmh_area), 1)
        
        c_peri_ss = np.zeros_like(c_peri_s)
        c_para_ss = np.zeros_like(c_peri_s)
        c_juxt_ss = np.zeros_like(c_peri_s)

        c_peri_ss[0:len(c_peri_s)] = c_peri_s / whole_area
        c_para_ss[0:len(c_para_s)] = c_para_s / whole_area
        c_juxt_ss[0:len(c_juxt_s)] = c_juxt_s / whole_area

    # ==================== SAVE COMPREHENSIVE SUMMARY JSON ====================
    # Create a comprehensive summary file with all key data
    comprehensive_summary = {
        "metadata": {
            "subject_id": id,
            "timepoint": tp,
            "analysis_type": "wmh_depth_analysis",
            "timestamp": datetime.now().isoformat(),
            "generated_plots": [
                f"All_Plaque_Depth_tp{tp}.png",
                f"Slices_Plaque_Depth_tp{tp}.png", 
                f"Category_Plaque_Depth_tp{tp}.png",
                f"Pie_Plaque_Depth_tp{tp}.png",
                f"BoxPlot_Plaque_Penetration_tp{tp}.png"
            ],
            "generated_data_files": [
                f"sorted_depth_scatter_data_tp{tp}.json",
                f"slice_wise_depth_scatter_data_tp{tp}.json",
                f"category_depth_grouped_bar_data_tp{tp}.json", 
                f"pie_chart_depth_distribution_data_tp{tp}.json",
                f"box_plot_depth_distribution_data_tp{tp}.json",
                f"comprehensive_depth_summary_tp{tp}.json"
            ]
        },
        "global_statistics": {
            "total_plaques": len(wmh_cob) if len(wmh_cob) > 0 else 0,
            "total_depth_mm": float(total_value) if not np.isnan(total_value) and not np.isinf(total_value) else 0.0,
            "depth_statistics": {
                "mean_mm": float(np.mean(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.mean(wmh_cob)) else 0.0,
                "median_mm": float(np.median(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.median(wmh_cob)) else 0.0,
                "std_mm": float(np.std(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.std(wmh_cob)) else 0.0,
                "min_mm": float(np.min(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.min(wmh_cob)) else 0.0,
                "max_mm": float(np.max(wmh_cob)) if len(wmh_cob) > 0 and not np.isnan(np.max(wmh_cob)) else 0.0
            }
        },
        "category_breakdown": {
            "periventricular": {
                "count": int(np.sum(np.array(wmh_code) == 1)) if len(wmh_code) > 0 else 0,
                "total_depth_mm": float(np.sum(c_peri_ss)) if len(c_peri_ss) > 0 and not np.isnan(np.sum(c_peri_ss)) else 0.0,
                "percentage_of_total": float((np.sum(c_peri_ss) / total_value * 100)) if total_value > 0 and not np.isnan(total_value) and not np.isinf(total_value) else 0.0
            },
            "paraventricular": {
                "count": int(np.sum(np.array(wmh_code) == 2)) if len(wmh_code) > 0 else 0,
                "total_depth_mm": float(np.sum(c_para_ss)) if len(c_para_ss) > 0 and not np.isnan(np.sum(c_para_ss)) else 0.0,
                "percentage_of_total": float((np.sum(c_para_ss) / total_value * 100)) if total_value > 0 and not np.isnan(total_value) and not np.isinf(total_value) else 0.0
            },
            "juxtacortical": {
                "count": int(np.sum(np.array(wmh_code) == 3)) if len(wmh_code) > 0 else 0,
                "total_depth_mm": float(np.sum(c_juxt_ss)) if len(c_juxt_ss) > 0 and not np.isnan(np.sum(c_juxt_ss)) else 0.0,
                "percentage_of_total": float((np.sum(c_juxt_ss) / total_value * 100)) if total_value > 0 and not np.isnan(total_value) and not np.isinf(total_value) else 0.0
            }
        },
        "slice_analysis": {
            "total_slices": len(wmh_num) if len(wmh_num) > 0 else 0,
            "slices_with_plaques": int(np.sum(np.array(wmh_cob_slice) > 0)) if len(wmh_cob_slice) > 0 else 0,
            "slice_depth_statistics": {
                "mean_depth_per_slice_mm": float(np.mean(wmh_cob_slice)) if len(wmh_cob_slice) > 0 and not np.isnan(np.mean(wmh_cob_slice)) else 0.0,
                "max_depth_per_slice_mm": float(np.max(wmh_cob_slice)) if len(wmh_cob_slice) > 0 and not np.isnan(np.max(wmh_cob_slice)) else 0.0,
                "std_depths_per_slice_mm": float(np.std(wmh_cob_slice)) if len(wmh_cob_slice) > 0 and not np.isnan(np.std(wmh_cob_slice)) else 0.0
            }
        },
        "raw_data": {
            "wmh_areas": wmh_area.tolist() if hasattr(wmh_area, 'tolist') and len(wmh_area) > 0 else (list(wmh_area) if len(wmh_area) > 0 else []),
            "wmh_cobs": wmh_cob.tolist() if hasattr(wmh_cob, 'tolist') and len(wmh_cob) > 0 else (list(wmh_cob) if len(wmh_cob) > 0 else []),
            "wmh_codes": list(wmh_code) if len(wmh_code) > 0 else [],
            "wmh_num_per_slice": list(wmh_num) if len(wmh_num) > 0 else [],
            "depth_per_slice": wmh_cob_slice.tolist() if hasattr(wmh_cob_slice, 'tolist') and len(wmh_cob_slice) > 0 else (list(wmh_cob_slice) if len(wmh_cob_slice) > 0 else []),
            "category_depth_per_slice": {
                "periventricular": c_peri_ss.tolist() if hasattr(c_peri_ss, 'tolist') else list(c_peri_ss),
                "paraventricular": c_para_ss.tolist() if hasattr(c_para_ss, 'tolist') else list(c_para_ss),
                "juxtacortical": c_juxt_ss.tolist() if hasattr(c_juxt_ss, 'tolist') else list(c_juxt_ss)
            }
        }
    }

    # Convert the entire dictionary to ensure all NumPy types are handled
    comprehensive_summary = convert_numpy_to_serializable(comprehensive_summary)

    # Save comprehensive summary with error handling
    try:
        with open(os.path.join(save_path, f'comprehensive_depth_summary_tp{tp}.json'), 'w') as f:
            json.dump(comprehensive_summary, f, indent=2)
        print(f"Successfully saved comprehensive summary to comprehensive_depth_summary_tp{tp}.json")
    except Exception as e:
        print(f"Error saving comprehensive summary: {e}")
        # Optionally save a simplified version or debug info
        print("Attempting to save simplified version...")
        try:
            # Create a simplified version with just the basic info
            simplified_summary = {
                "metadata": comprehensive_summary["metadata"],
                "global_statistics": {
                    "total_plaques": comprehensive_summary["global_statistics"]["total_plaques"],
                    "total_depth_mm": comprehensive_summary["global_statistics"]["total_depth_mm"]
                },
                "error_info": {
                    "original_error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
            with open(os.path.join(save_path, f'simplified_depth_summary_tp{tp}.json'), 'w') as f:
                json.dump(simplified_summary, f, indent=2)
            print("Successfully saved simplified summary")
        except Exception as e2:
            print(f"Failed to save even simplified summary: {e2}")

    # # Set periventricular to zero as per original logic
    # c_peri_ss[:] = 0

    # Update metadata
    depth_general_data = {
        "depth_of_all_found_WMH": list(np.uint16(wmh_cob)) if len(wmh_cob) > 0 else [],
        "whole_brain_total_depth": int(total_value)
    }
    depth_periventricular_data = {
        "whole_brain_total_depth": int(np.sum(c_peri_ss))
    }
    depth_paraventricular_data = {
        "whole_brain_total_depth": int(np.sum(c_para_ss))
    }
    depth_juxtacortical_data = {
        "whole_brain_total_depth": int(np.sum(c_juxt_ss))
    }

    # Note: These function calls will need to be uncommented if the functions exist
    update_wmh_data(wmh_m_data, "depth", "general", depth_general_data)
    update_wmh_data(wmh_m_data, "depth", "periventricular", depth_periventricular_data)
    update_wmh_data(wmh_m_data, "depth", "paraventricular", depth_paraventricular_data)
    update_wmh_data(wmh_m_data, "depth", "juxtacortical", depth_juxtacortical_data)

    return [c_peri_ss, c_para_ss, c_juxt_ss]

#
def convert_numpy_to_serializable(obj):
    """
    Recursively convert NumPy arrays and data types to JSON-serializable formats
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_serializable(item) for item in obj)
    return obj

#
def hitmap_2d_subj(save_path, mask_wmh, brain_mask, voxel_size, spec_id=[], tp=0):
    # print(f'\n\n {np.sum(mask_wmh)*voxel_size[0]*voxel_size[1]} \n\n')
    wmh = []
    wmh_data = []
    vox = []

    subject = str(spec_id)
    wmh_data = mask_wmh
    wmh_data[np.isnan(wmh_data)] = 0

    # superimposing the slices of each subject
    w, n, h = wmh_data.shape
    summ = np.zeros((w, n))
    for i in range(0, h):
        summ += wmh_data[..., i]

    brain_t = np.zeros((w, n))

    # pixel_spacing of the subject
    voxel_size = voxel_size[0:2]  # only Px, Py
    vox.append(voxel_size)

    # translating 2d projection (sum image) to get its brain center aligned with a point fixed for all subjects
    dx = 0
    dy = 0
    translation_matrix = np.float32([[1, 0, dx],
                                     [0, 1, dy]])

    # Apply the translation using warpAffine function
    translated_image = cv2.warpAffine(summ, translation_matrix, (summ.shape[1], summ.shape[0]))
    summ_t = translated_image
    summ_t = summ

    # making an averaged brain tissue mask
    brain_mask = brain_tissue(brain_mask)  # getting a 2d mask of 3d masks

    # Apply the translation using warpAffine function
    brain_image = cv2.warpAffine(brain_mask, translation_matrix, (n, w))
    brain_t += brain_image

    # brain_t = np.where(brain_t > 220/255, 1, 0)
    brain_t = brain_t[13:-13, 13:-13]
    shapee = np.shape(brain_t)
    temp = cv2.resize(brain_t, (2 * shapee[0], 2 * shapee[1]))
    brain_t = temp

    # brain_t = skimage.transform.rotate(brain_t, -90)
    brain_t = (brain_t / np.max(brain_t)) * 255

    #
    summ_t = summ_t[13:-13, 13:-13]
    shapee = np.shape(summ_t)
    temp = cv2.resize(summ_t, (2 * shapee[0], 2 * shapee[1]))
    summ_t = temp

    # summ_t = skimage.transform.rotate(summ_t, -90)
    summ_t = (summ_t / max(np.max(summ_t), 1)) * 255

    # plt.figure('Brain Map')
    # plt.imshow(brain_t)
    # plt.show()

    # fig, axl = plt.subplots(figsize=((2.5 * np.shape(summ_t)[0])/100, (2 * np.shape(summ_t)[0])/100))
    fig, axl = plt.subplots(figsize=(615/100, 770/100))
    # Define font properties
    font_properties = {
        'family': 'serif',  # Font family
        'size': 16,  # Font size
        'weight': 'bold',  # Font weight
        # 'style': 'italic'  # Font style
    }
    # fig.suptitle(f'2D  HIT  MAP  of  Subject : {spec_id}', fontdict=font_properties)

    axl.imshow(summ_t, cmap='plasma')
    # plt.colorbar(axl.imshow(summ_t, cmap='plasma'))
    axl.imshow(brain_t, alpha=0.15)
    axl.axis('off')

    # saving HitMap:
    hit_img_name = os.path.join(save_path, 'HitMap_tp' + str(tp) + '.png')
    plt.savefig(hit_img_name)
    plt.close()
    hit_image = skimage.io.imread(hit_img_name)
    hit_image = hit_image[155:635, 80:550]
    skimage.io.imsave(hit_img_name, hit_image)

    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

    return

#
def brain_tissue(brain_mask_):
    kernel4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (50, 50))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))

    # vents_m = cv2.morphologyEx(vents_m, cv2.MORPH_OPEN, kernel3)
    brain_mask_ = cv2.morphologyEx(brain_mask_, cv2.MORPH_OPEN, kernel3)
    brain_mask_ = cv2.morphologyEx(brain_mask_, cv2.MORPH_CLOSE, kernel4)
    brain_mask_ = cv2.morphologyEx(brain_mask_, cv2.MORPH_CLOSE, kernel5)

    cxx = int(brain_mask_.shape[0] / 2)
    cyy = int(brain_mask_.shape[1] / 2)
    b_mask = np.zeros((brain_mask_.shape[0], brain_mask_.shape[1]))
    for w in range(0, brain_mask_.shape[2]):
        b_mask += brain_mask_[..., w]
    b_mask = np.where(b_mask > 0, 1, 0).astype(np.uint8)

    return b_mask

