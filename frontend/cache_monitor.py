"""
Cache Monitoring and Health Checks for Trading Application
Provides visibility into cache performance and health metrics
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from cache_manager import frontend_cache

logger = logging.getLogger(__name__)

class CacheMonitor:
    """
    Monitor cache performance and health
    Tracks hit rates, memory usage, and provides health checks
    """
    
    def __init__(self):
        self.monitoring_enabled = True
        self.performance_history = []
        self.max_history_size = 1000
        
    def get_cache_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cache health report
        """
        stats = frontend_cache.get_stats()
        
        # Calculate performance metrics
        hit_rate = stats.get('hit_rate', 0)
        total_requests = stats.get('total_requests', 0)
        total_entries = stats.get('total_entries', 0)
        max_size = stats.get('max_size', 0)
        evictions = stats.get('evictions', 0)
        
        # Determine health status
        health_status = self._calculate_health_status(stats)
        
        # Memory efficiency
        memory_usage_percent = (total_entries / max_size * 100) if max_size > 0 else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': health_status,
            'performance': {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'cache_efficiency': self._get_efficiency_rating(hit_rate)
            },
            'memory': {
                'total_entries': total_entries,
                'max_size': max_size,
                'usage_percent': round(memory_usage_percent, 2),
                'memory_status': self._get_memory_status(memory_usage_percent)
            },
            'evictions': {
                'total_evictions': evictions,
                'eviction_rate': round(evictions / max(total_requests, 1) * 100, 2),
                'eviction_status': self._get_eviction_status(evictions, total_requests)
            },
            'recommendations': self._generate_recommendations(stats)
        }
        
        return report
    
    def _calculate_health_status(self, stats: Dict[str, Any]) -> str:
        """Calculate overall cache health status"""
        hit_rate = stats.get('hit_rate', 0)
        total_entries = stats.get('total_entries', 0)
        max_size = stats.get('max_size', 0)
        evictions = stats.get('evictions', 0)
        total_requests = stats.get('total_requests', 0)
        
        # Calculate scores
        hit_rate_score = min(hit_rate / 70, 1.0)  # 70% hit rate is good
        memory_score = 1.0 - (total_entries / max_size) if max_size > 0 else 1.0
        eviction_score = 1.0 - min(evictions / max(total_requests, 1), 0.1) * 10
        
        overall_score = (hit_rate_score + memory_score + eviction_score) / 3
        
        if overall_score >= 0.8:
            return "EXCELLENT"
        elif overall_score >= 0.6:
            return "GOOD"
        elif overall_score >= 0.4:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_efficiency_rating(self, hit_rate: float) -> str:
        """Get efficiency rating based on hit rate"""
        if hit_rate >= 80:
            return "EXCELLENT"
        elif hit_rate >= 60:
            return "GOOD"
        elif hit_rate >= 40:
            return "FAIR"
        else:
            return "POOR"
    
    def _get_memory_status(self, usage_percent: float) -> str:
        """Get memory usage status"""
        if usage_percent >= 90:
            return "CRITICAL"
        elif usage_percent >= 75:
            return "HIGH"
        elif usage_percent >= 50:
            return "MODERATE"
        else:
            return "LOW"
    
    def _get_eviction_status(self, evictions: int, total_requests: int) -> str:
        """Get eviction status"""
        if total_requests == 0:
            return "NONE"
        
        eviction_rate = evictions / max(1, total_requests) * 100
        
        if eviction_rate >= 10:
            return "HIGH"
        elif eviction_rate >= 5:
            return "MODERATE"
        elif eviction_rate >= 1:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        hit_rate = stats.get('hit_rate', 0)
        total_entries = stats.get('total_entries', 0)
        max_size = stats.get('max_size', 0)
        evictions = stats.get('evictions', 0)
        
        # Hit rate recommendations
        if hit_rate < 40:
            recommendations.append("Low hit rate detected. Consider increasing TTL values for stable data.")
        elif hit_rate > 90:
            recommendations.append("Very high hit rate. Monitor data freshness for trading accuracy.")
        
        # Memory recommendations
        memory_usage = (total_entries / max_size * 100) if max_size > 0 else 0
        if memory_usage > 85:
            recommendations.append("High memory usage. Consider increasing cache size or reducing TTL values.")
        elif memory_usage < 20:
            recommendations.append("Low memory usage. Cache size might be oversized.")
        
        # Eviction recommendations
        if evictions > 100:
            recommendations.append("High eviction rate. Consider increasing cache size.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Cache performance is optimal. No immediate action required.")
        
        return recommendations
    
    def log_cache_performance(self):
        """Log current cache performance metrics"""
        if not self.monitoring_enabled:
            return
        
        stats = frontend_cache.get_stats()
        health_report = self.get_cache_health_report()
        
        logger.info(
            f"Cache Performance - Hit Rate: {stats.get('hit_rate', 0):.1f}%, "
            f"Entries: {stats.get('total_entries', 0)}, "
            f"Health: {health_report['overall_health']}"
        )
        
        # Store in history for trending
        self.performance_history.append({
            'timestamp': datetime.now(),
            'hit_rate': stats.get('hit_rate', 0),
            'total_entries': stats.get('total_entries', 0),
            'evictions': stats.get('evictions', 0)
        })
        
        # Limit history size
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
    
    def get_performance_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance trends over specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_history = [
            entry for entry in self.performance_history
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_history:
            return {'error': 'No performance data available for the specified period'}
        
        # Calculate trends
        hit_rates = [entry['hit_rate'] for entry in recent_history]
        entry_counts = [entry['total_entries'] for entry in recent_history]
        eviction_counts = [entry['evictions'] for entry in recent_history]
        
        return {
            'period_hours': hours,
            'data_points': len(recent_history),
            'hit_rate_trend': {
                'average': sum(hit_rates) / max(1, len(hit_rates)),
                'min': min(hit_rates) if hit_rates else 0,
                'max': max(hit_rates) if hit_rates else 0,
                'current': hit_rates[-1] if hit_rates else 0
            },
            'cache_size_trend': {
                'average': sum(entry_counts) / max(1, len(entry_counts)),
                'min': min(entry_counts) if entry_counts else 0,
                'max': max(entry_counts) if entry_counts else 0,
                'current': entry_counts[-1] if entry_counts else 0
            },
            'evictions_trend': {
                'total': eviction_counts[-1] if eviction_counts else 0,
                'rate_per_hour': (eviction_counts[-1] - eviction_counts[0]) / hours if len(eviction_counts) > 1 else 0
            }
        }
    
    def check_cache_alerts(self) -> List[Dict[str, Any]]:
        """Check for cache-related alerts"""
        alerts = []
        stats = frontend_cache.get_stats()
        
        # Hit rate alert
        hit_rate = stats.get('hit_rate', 0)
        if hit_rate < 30:
            alerts.append({
                'severity': 'HIGH',
                'type': 'LOW_HIT_RATE',
                'message': f"Cache hit rate is very low: {hit_rate:.1f}%",
                'recommendation': 'Check TTL settings and data access patterns'
            })
        
        # Memory usage alert
        total_entries = stats.get('total_entries', 0)
        max_size = stats.get('max_size', 0)
        if max_size > 0 and (total_entries / max_size) > 0.9:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'HIGH_MEMORY_USAGE',
                'message': f"Cache memory usage is high: {total_entries}/{max_size} entries",
                'recommendation': 'Consider increasing cache size or reducing TTL values'
            })
        
        # Eviction rate alert
        evictions = stats.get('evictions', 0)
        total_requests = stats.get('total_requests', 1)
        eviction_rate = evictions / max(1, total_requests) * 100
        if eviction_rate > 15:
            alerts.append({
                'severity': 'MEDIUM',
                'type': 'HIGH_EVICTION_RATE',
                'message': f"High cache eviction rate: {eviction_rate:.1f}%",
                'recommendation': 'Increase cache size or optimize data access patterns'
            })
        
        return alerts
    
    def enable_monitoring(self):
        """Enable cache monitoring"""
        self.monitoring_enabled = True
        logger.info("Cache monitoring enabled")
    
    def disable_monitoring(self):
        """Disable cache monitoring"""
        self.monitoring_enabled = False
        logger.info("Cache monitoring disabled")


# Global cache monitor instance
cache_monitor = CacheMonitor()

def get_cache_health_dashboard() -> Dict[str, Any]:
    """Get comprehensive cache health dashboard data"""
    health_report = cache_monitor.get_cache_health_report()
    alerts = cache_monitor.check_cache_alerts()
    trends = cache_monitor.get_performance_trends()
    
    return {
        'health_report': health_report,
        'alerts': alerts,
        'trends': trends,
        'last_updated': datetime.now().isoformat()
    }

def log_cache_performance():
    """Convenience function to log cache performance"""
    cache_monitor.log_cache_performance()