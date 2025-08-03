"""
Visualization utilities for cell classification analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """Generate visualizations for cell classification analysis results."""
    
    def __init__(self, classification_mode: str):
        self.classification_mode = classification_mode
        
    def create_enhanced_visualizations(self, analysis_results: Dict, output_dir: Path) -> None:
        """Create enhanced visualization plots including relative analysis and probability densities."""
        logger.info("Creating enhanced visualization plots...")
        
        basic_stats = analysis_results.get('basic_statistics', {})
        if not basic_stats.get('prediction_counts'):
            logger.warning("No prediction data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(26, 32))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Enhanced Cell Classification Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: Overall prediction distribution (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        counts = basic_stats['prediction_counts']
        labels = list(counts.keys())
        sizes = list(counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Overall Prediction Distribution', fontweight='bold')
        
        # Plot 2: Relative distribution excluding uncertain (pie chart)
        ax2 = fig.add_subplot(gs[0, 1])
        decided_counts = basic_stats.get('decided_prediction_counts', {})
        if decided_counts:
            rel_labels = list(decided_counts.keys())
            rel_sizes = list(decided_counts.values())
            rel_colors = plt.cm.Dark2(np.linspace(0, 1, len(rel_labels)))
            
            wedges, texts, autotexts = ax2.pie(rel_sizes, labels=rel_labels, autopct='%1.1f%%', 
                                              colors=rel_colors, startangle=90)
            ax2.set_title('Relative Distribution\n(Excluding Uncertain)', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No decided\npredictions', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Relative Distribution\n(Excluding Uncertain)', fontweight='bold')
        
        # Plot 3: Comparison bar chart
        ax3 = fig.add_subplot(gs[0, 2])
        all_predictions = list(counts.keys())
        overall_pcts = [basic_stats.get('prediction_percentages', {}).get(pred, 0) for pred in all_predictions]
        relative_pcts = [basic_stats.get('relative_percentages_excluding_uncertain', {}).get(pred, 0) for pred in all_predictions]
        
        x = np.arange(len(all_predictions))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, overall_pcts, width, label='Overall %', alpha=0.8)
        bars2 = ax3.bar(x + width/2, relative_pcts, width, label='Relative % (decided only)', alpha=0.8)
        
        ax3.set_xlabel('Prediction Type')
        ax3.set_ylabel('Percentage')
        ax3.set_title('Overall vs Relative Percentages', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(all_predictions, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Processing statistics
        ax4 = fig.add_subplot(gs[1, 0])
        if 'processing_metadata' in analysis_results:
            proc_meta = analysis_results['processing_metadata']
            success_count = proc_meta.get('successfully_processed', 0)
            error_count = proc_meta.get('processing_errors', 0)
            
            proc_labels = ['Successful', 'Errors']
            proc_counts = [success_count, error_count]
            proc_colors = ['green', 'red']
            
            bars = ax4.bar(proc_labels, proc_counts, color=proc_colors, alpha=0.7)
            ax4.set_title('Processing Success Rate', fontweight='bold')
            ax4.set_ylabel('Number of Images')
            
            # Add percentage labels
            total_proc = success_count + error_count
            if total_proc > 0:
                for bar, count in zip(bars, proc_counts):
                    pct = count / total_proc * 100
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(proc_counts)*0.01,
                            f'{pct:.1f}%', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'Processing metadata\nnot available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Processing Statistics', fontweight='bold')
        
        # Plot 5: Confidence levels
        ax5 = fig.add_subplot(gs[1, 1])
        if 'confidence_statistics' in analysis_results:
            conf_stats = analysis_results['confidence_statistics'].get('confidence_levels', {})
            if conf_stats:
                conf_labels = ['High (≥0.8)', 'Medium (0.5-0.8)', 'Low (<0.5)']
                conf_counts = [
                    conf_stats.get('high_confidence', {}).get('count', 0),
                    conf_stats.get('medium_confidence', {}).get('count', 0),
                    conf_stats.get('low_confidence', {}).get('count', 0)
                ]
                conf_colors = ['green', 'orange', 'red']
                
                bars = ax5.bar(conf_labels, conf_counts, color=conf_colors, alpha=0.7)
                ax5.set_title('Confidence Level Distribution', fontweight='bold')
                ax5.set_ylabel('Number of Cells')
                ax5.tick_params(axis='x', rotation=45)
                
                # Add count labels
                for bar, count in zip(bars, conf_counts):
                    if count > 0:
                        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(conf_counts)*0.01,
                                str(count), ha='center', va='bottom')
            else:
                ax5.text(0.5, 0.5, 'Confidence statistics\nnot available', 
                        transform=ax5.transAxes, ha='center', va='center')
                ax5.set_title('Confidence Distribution', fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'Confidence statistics\nnot available', 
                    transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title('Confidence Distribution', fontweight='bold')
        
        # Plot 6: Decision certainty (decided vs uncertain)
        ax6 = fig.add_subplot(gs[1, 2])
        decided_count = basic_stats.get('decided_predictions_count', 0)
        uncertain_count = counts.get('uncertain', 0)
        total_count = basic_stats.get('total_images_analyzed', 0)
        
        decision_labels = ['Decided', 'Uncertain']
        decision_counts = [decided_count, uncertain_count]
        decision_colors = ['lightblue', 'lightcoral']
        
        bars = ax6.bar(decision_labels, decision_counts, color=decision_colors, alpha=0.7)
        ax6.set_title('Decision Certainty', fontweight='bold')
        ax6.set_ylabel('Number of Cells')
        
        # Add percentage labels
        if total_count > 0:
            for bar, count in zip(bars, decision_counts):
                pct = count / total_count * 100
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(decision_counts)*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        # Create probability density plots based on classification mode
        if self.classification_mode == "binary":
            self._add_binary_probability_plots(fig, gs, analysis_results)
        else:
            self._add_ternary_probability_plots(fig, gs, analysis_results)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for suptitle
        
        # Save the plot
        plot_path = output_dir / f"{self.classification_mode}_enhanced_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        logger.info(f"Enhanced visualization saved to {plot_path}")
    
    def _add_binary_probability_plots(self, fig, gs, analysis: Dict) -> None:
        """Add binary classification probability density plots."""
        detailed_analysis = analysis.get('detailed_analysis', {})
        
        # Plot 7: Probability density for all predictions
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Extract probability data
        cancerous_stats = detailed_analysis.get('cancerous_cells', {}).get('stats', {})
        non_cancerous_stats = detailed_analysis.get('non_cancerous_cells', {}).get('stats', {})
        uncertain_stats = detailed_analysis.get('uncertain_cells', {}).get('stats', {})
        
        # Create synthetic data for visualization (since we don't have raw probabilities)
        # This is an approximation based on the statistics
        x = np.linspace(0, 1, 100)
        
        # Plot density curves if we have statistics
        if cancerous_stats.get('count', 0) > 0:
            mean_canc = cancerous_stats.get('mean', 0.8)
            std_canc = max(cancerous_stats.get('std', 0.1), 0.01)  # Prevent division by zero
            y_canc = self._gaussian_kde(x, mean_canc, std_canc)
            ax7.plot(x, y_canc, label=f'Cancerous (n={cancerous_stats.get("count", 0)})', 
                    color='red', linewidth=2)
        
        if non_cancerous_stats.get('count', 0) > 0:
            mean_non_canc = non_cancerous_stats.get('mean', 0.2)
            std_non_canc = max(non_cancerous_stats.get('std', 0.1), 0.01)
            y_non_canc = self._gaussian_kde(x, mean_non_canc, std_non_canc)
            ax7.plot(x, y_non_canc, label=f'Non-Cancerous (n={non_cancerous_stats.get("count", 0)})', 
                    color='blue', linewidth=2)
        
        if uncertain_stats.get('count', 0) > 0:
            mean_unc = uncertain_stats.get('mean', 0.5)
            std_unc = max(uncertain_stats.get('std', 0.1), 0.01)
            y_unc = self._gaussian_kde(x, mean_unc, std_unc)
            ax7.plot(x, y_unc, label=f'Uncertain (n={uncertain_stats.get("count", 0)})', 
                    color='gray', linewidth=2, linestyle='--')
        
        # Add threshold lines if thresholds are available
        try:
            ax7.axvline(x=self.classifier.confidence_threshold_low, color='green', linestyle='--', 
                       alpha=0.7, label=f'Low Threshold ({self.classifier.confidence_threshold_low:.2f})')
            ax7.axvline(x=self.classifier.confidence_threshold_high, color='red', linestyle='--', 
                       alpha=0.7, label=f'High Threshold ({self.classifier.confidence_threshold_high:.2f})')
        except AttributeError:
            # Use default thresholds if not available
            ax7.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Low Threshold (0.3)')
            ax7.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Threshold (0.7)')
        
        ax7.axvline(x=0.5, color='black', linestyle=':', alpha=0.5, label='0.5 Threshold')
        
        ax7.set_xlabel('Predicted Probability')
        ax7.set_ylabel('Density')
        ax7.set_title('Probability Density Distribution (Binary Classification)', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Decided-only probability distribution
        ax8 = fig.add_subplot(gs[3, :2])
        
        decided_stats = detailed_analysis.get('decided_only_analysis', {})
        if decided_stats.get('count', 0) > 0:
            # Plot decided predictions only
            if cancerous_stats.get('count', 0) > 0:
                ax8.plot(x, y_canc, label=f'Cancerous (decided)', color='red', linewidth=2)
            if non_cancerous_stats.get('count', 0) > 0:
                ax8.plot(x, y_non_canc, label=f'Non-Cancerous (decided)', color='blue', linewidth=2)
            
            try:
                ax8.axvline(x=self.classifier.confidence_threshold_low, color='green', linestyle='--', 
                           alpha=0.7, label=f'Low Threshold ({self.classifier.confidence_threshold_low:.2f})')
                ax8.axvline(x=self.classifier.confidence_threshold_high, color='red', linestyle='--', 
                           alpha=0.7, label=f'High Threshold ({self.classifier.confidence_threshold_high:.2f})')
            except AttributeError:
                ax8.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Low Threshold (0.3)')
                ax8.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Threshold (0.7)')
            
            rel_canc_pct = decided_stats.get('relative_cancerous_percentage', 0)
            rel_non_canc_pct = decided_stats.get('relative_non_cancerous_percentage', 0)
            
            ax8.set_title(f'Decided Predictions Only\n(Cancerous: {rel_canc_pct:.1f}%, Non-Cancerous: {rel_non_canc_pct:.1f}%)', 
                         fontweight='bold')
        else:
            ax8.text(0.5, 0.5, 'No decided predictions\navailable', ha='center', va='center', 
                    transform=ax8.transAxes, fontsize=12)
            ax8.set_title('Decided Predictions Only', fontweight='bold')
        
        ax8.set_xlabel('Predicted Probability')
        ax8.set_ylabel('Density')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Summary statistics
        ax9 = fig.add_subplot(gs[2:, 2])
        
        # Create text summary
        basic_stats = analysis.get('basic_statistics', {})
        insights = analysis.get('insights_and_recommendations', {}).get('summary', {})
        
        summary_text = f"""BINARY CLASSIFICATION SUMMARY

            Total Images: {basic_stats.get('total_images_analyzed', 'N/A')}
            Success Rate: {basic_stats.get('success_rate', 0):.1f}%

            OVERALL DISTRIBUTION:
            Cancerous: {basic_stats.get('prediction_counts', {}).get('cancerous', 0)} ({basic_stats.get('prediction_percentages', {}).get('cancerous', 0):.1f}%)
            Non-Cancerous: {basic_stats.get('prediction_counts', {}).get('non-cancerous', 0)} ({basic_stats.get('prediction_percentages', {}).get('non-cancerous', 0):.1f}%)
            Uncertain: {basic_stats.get('prediction_counts', {}).get('uncertain', 0)} ({basic_stats.get('prediction_percentages', {}).get('uncertain', 0):.1f}%)

            DECIDED PREDICTIONS ONLY:
            Total Decided: {basic_stats.get('decided_predictions_count', 0)} ({basic_stats.get('decided_predictions_percentage', 0):.1f}%)
            Relative Cancerous: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('cancerous', 0):.1f}%
            Relative Non-Cancerous: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('non-cancerous', 0):.1f}%

            CONFIDENCE ASSESSMENT:
            {insights.get('confidence_assessment', 'N/A')}

            PRIMARY FINDING:
            {insights.get('primary_finding', 'N/A')}"""
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax9.set_title('Analysis Summary', fontweight='bold')
        ax9.axis('off')
    
    def _add_ternary_probability_plots(self, fig, gs, analysis: Dict) -> None:
        """Add ternary classification probability density plots."""
        detailed_analysis = analysis.get('detailed_analysis', {})
        
        # Plot 7: Confidence distribution
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Extract confidence data
        overall_conf_stats = detailed_analysis.get('overall_confidence_distribution', {})
        decided_conf_stats = detailed_analysis.get('decided_only_analysis', {}).get('confidence_distribution', {})
        
        # Create synthetic confidence distribution
        x = np.linspace(0, 1, 100)
        
        if overall_conf_stats.get('count', 0) > 0:
            mean_conf = overall_conf_stats.get('mean', 0.5)
            std_conf = max(overall_conf_stats.get('std', 0.2), 0.01)
            y_conf = self._gaussian_kde(x, mean_conf, std_conf)
            ax7.plot(x, y_conf, label=f'All Predictions (n={overall_conf_stats.get("count", 0)})', 
                    color='purple', linewidth=2)
        
        if decided_conf_stats.get('count', 0) > 0:
            mean_dec_conf = decided_conf_stats.get('mean', 0.6)
            std_dec_conf = max(decided_conf_stats.get('std', 0.15), 0.01)
            y_dec_conf = self._gaussian_kde(x, mean_dec_conf, std_dec_conf)
            ax7.plot(x, y_dec_conf, label=f'Decided Only (n={decided_conf_stats.get("count", 0)})', 
                    color='darkblue', linewidth=2, linestyle='--')
        
        # Add threshold line
        try:
            ax7.axvline(x=self.classifier.uncertainty_threshold, color='red', linestyle='--', 
                       alpha=0.7, label=f'Uncertainty Threshold ({self.classifier.uncertainty_threshold:.2f})')
        except AttributeError:
            ax7.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Uncertainty Threshold (0.5)')
        
        ax7.set_xlabel('Confidence Level')
        ax7.set_ylabel('Density')
        ax7.set_title('Confidence Distribution (Ternary Classification)', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Class distribution for decided predictions
        ax8 = fig.add_subplot(gs[3, :2])
        
        # Get class statistics
        canc_stats = detailed_analysis.get('cancerous_cells', {})
        non_canc_stats = detailed_analysis.get('non_cancerous_cells', {})
        fp_stats = detailed_analysis.get('false_positive_cells', {})
        
        classes = ['Cancerous', 'Non-Cancerous', 'False-Positive']
        decided_counts = [
            canc_stats.get('count', 0),
            non_canc_stats.get('count', 0),
            fp_stats.get('count', 0)
        ]
        relative_pcts = [
            canc_stats.get('relative_percentage', 0),
            non_canc_stats.get('relative_percentage', 0),
            fp_stats.get('relative_percentage', 0)
        ]
        
        x_pos = np.arange(len(classes))
        bars = ax8.bar(x_pos, decided_counts, color=['red', 'blue', 'orange'], alpha=0.7)
        
        # Add percentage labels
        for bar, pct in zip(bars, relative_pcts):
            height = bar.get_height()
            if height > 0:
                ax8.text(bar.get_x() + bar.get_width()/2., height + max(decided_counts)*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        ax8.set_xlabel('Prediction Class')
        ax8.set_ylabel('Count')
        ax8.set_title('Decided Predictions by Class', fontweight='bold')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(classes)
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Summary statistics
        ax9 = fig.add_subplot(gs[2:, 2])
        
        # Create text summary
        basic_stats = analysis.get('basic_statistics', {})
        insights = analysis.get('insights_and_recommendations', {}).get('summary', {})
        
        summary_text = f"""TERNARY CLASSIFICATION SUMMARY

            Total Images: {basic_stats.get('total_images_analyzed', 'N/A')}
            Success Rate: {basic_stats.get('success_rate', 0):.1f}%

            OVERALL DISTRIBUTION:
            Cancerous: {basic_stats.get('prediction_counts', {}).get('cancerous', 0)} ({basic_stats.get('prediction_percentages', {}).get('cancerous', 0):.1f}%)
            Non-Cancerous: {basic_stats.get('prediction_counts', {}).get('non-cancerous', 0)} ({basic_stats.get('prediction_percentages', {}).get('non-cancerous', 0):.1f}%)
            False-Positive: {basic_stats.get('prediction_counts', {}).get('false-positive', 0)} ({basic_stats.get('prediction_percentages', {}).get('false-positive', 0):.1f}%)
            Uncertain: {basic_stats.get('prediction_counts', {}).get('uncertain', 0)} ({basic_stats.get('prediction_percentages', {}).get('uncertain', 0):.1f}%)

            DECIDED PREDICTIONS ONLY:
            Total Decided: {basic_stats.get('decided_predictions_count', 0)} ({basic_stats.get('decided_predictions_percentage', 0):.1f}%)
            Relative Cancerous: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('cancerous', 0):.1f}%
            Relative Non-Cancerous: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('non-cancerous', 0):.1f}%
            Relative False-Positive: {basic_stats.get('relative_percentages_excluding_uncertain', {}).get('false-positive', 0):.1f}%

            CONFIDENCE ASSESSMENT:
            {insights.get('confidence_assessment', 'N/A')}

            PRIMARY FINDING:
            {insights.get('primary_finding', 'N/A')}"""
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax9.set_title('Analysis Summary', fontweight='bold')
        ax9.axis('off')
    
    def _gaussian_kde(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Create a simple Gaussian kernel density estimate."""
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))