import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Configuración general de los gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Cargar datos
print("Cargando dataset...")
df = pd.read_csv('data/dengue_unified_dataset_enhanced.csv')
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# Crear directorio para guardar las imágenes
import os
if not os.path.exists('eda_plots'):
    os.makedirs('eda_plots')

# 1. Distribución temporal de casos por año
plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)
yearly_cases = df.groupby('year')['target_cases'].sum()
bars = plt.bar(yearly_cases.index, yearly_cases.values, 
               color=sns.color_palette("viridis", len(yearly_cases)))
plt.title('Casos Totales de Dengue por Año', fontsize=14, fontweight='bold')
plt.xlabel('Año')
plt.ylabel('Casos Totales')
for i, v in enumerate(yearly_cases.values):
    plt.text(yearly_cases.index[i], v + 10, str(v), ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 2. Casos por semana del año (patrón estacional)
plt.subplot(2, 2, 2)
weekly_cases = df.groupby('week')['target_cases'].sum()
plt.plot(weekly_cases.index, weekly_cases.values, marker='o', linewidth=2, markersize=4)
plt.title('Patrón Estacional - Casos por Semana del Año', fontsize=14, fontweight='bold')
plt.xlabel('Semana del Año')
plt.ylabel('Casos Totales')
plt.grid(True, alpha=0.3)
# Destacar la temporada alta (semanas 10-25)
plt.axvspan(10, 25, alpha=0.2, color='red', label='Temporada Alta')
plt.legend()

# 3. Top departamentos con más casos
plt.subplot(2, 2, 3)
dept_cases = df.groupby('departament_name')['target_cases'].sum().sort_values(ascending=False).head(10)
plt.barh(range(len(dept_cases)), dept_cases.values, color=sns.color_palette("rocket", len(dept_cases)))
plt.yticks(range(len(dept_cases)), dept_cases.index)
plt.title('Top 10 Departamentos con Más Casos', fontsize=14, fontweight='bold')
plt.xlabel('Casos Totales')
for i, v in enumerate(dept_cases.values):
    plt.text(v + 5, i, str(v), va='center', fontweight='bold')
plt.gca().invert_yaxis()

# 4. Distribución de casos por provincia
plt.subplot(2, 2, 4)
province_cases = df.groupby('province')['target_cases'].sum()
plt.pie(province_cases.values, labels=province_cases.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribución de Casos por Provincia', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots/1_temporal_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Análisis de variables climáticas
plt.figure(figsize=(16, 10))

# Heatmap de correlación
plt.subplot(2, 3, 1)
climate_vars = ['prec_2weeks', 'temp_2weeks_avg', 'humd_2weeks_avg', 'target_cases', 'trends_dengue']
corr_matrix = df[climate_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, cbar_kws={'shrink': 0.8})
plt.title('Matriz de Correlación - Variables Climáticas', fontsize=12, fontweight='bold')

# Distribución de temperatura
plt.subplot(2, 3, 2)
plt.hist(df['temp_2weeks_avg'].dropna(), bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.axvline(df['temp_2weeks_avg'].mean(), color='red', linestyle='--', 
            label=f'Media: {df["temp_2weeks_avg"].mean():.1f}°C')
plt.title('Distribución de Temperatura Promedio', fontsize=12, fontweight='bold')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(alpha=0.3)

# Distribución de humedad
plt.subplot(2, 3, 3)
plt.hist(df['humd_2weeks_avg'].dropna(), bins=30, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(df['humd_2weeks_avg'].mean(), color='red', linestyle='--', 
            label=f'Media: {df["humd_2weeks_avg"].mean():.1f}%')
plt.title('Distribución de Humedad Promedio', fontsize=12, fontweight='bold')
plt.xlabel('Humedad (%)')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(alpha=0.3)

# Precipitación vs Casos
plt.subplot(2, 3, 4)
# Crear bins para precipitación
df['prec_bins'] = pd.cut(df['prec_2weeks'], bins=5, labels=['Muy Baja', 'Baja', 'Media', 'Alta', 'Muy Alta'])
prec_cases = df.groupby('prec_bins', observed=True)['target_cases'].mean()
bars = plt.bar(range(len(prec_cases)), prec_cases.values, 
               color=sns.color_palette("Blues_r", len(prec_cases)))
plt.xticks(range(len(prec_cases)), prec_cases.index, rotation=45)
plt.title('Casos Promedio por Nivel de Precipitación', fontsize=12, fontweight='bold')
plt.ylabel('Casos Promedio')
for i, v in enumerate(prec_cases.values):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')

# Temperatura vs Casos
plt.subplot(2, 3, 5)
df['temp_bins'] = pd.cut(df['temp_2weeks_avg'], bins=5, labels=['Muy Fría', 'Fría', 'Templada', 'Caliente', 'Muy Caliente'])
temp_cases = df.groupby('temp_bins', observed=True)['target_cases'].mean()
bars = plt.bar(range(len(temp_cases)), temp_cases.values, 
               color=sns.color_palette("Reds", len(temp_cases)))
plt.xticks(range(len(temp_cases)), temp_cases.index, rotation=45)
plt.title('Casos Promedio por Nivel de Temperatura', fontsize=12, fontweight='bold')
plt.ylabel('Casos Promedio')
for i, v in enumerate(temp_cases.values):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')

# Trends vs Casos
plt.subplot(2, 3, 6)
plt.scatter(df['trends_dengue'], df['target_cases'], alpha=0.6, s=20)
z = np.polyfit(df['trends_dengue'].dropna(), df[df['trends_dengue'].notna()]['target_cases'], 1)
p = np.poly1d(z)
plt.plot(df['trends_dengue'].dropna().sort_values(), 
         p(df['trends_dengue'].dropna().sort_values()), "r--", alpha=0.8, linewidth=2)
plt.title('Google Trends vs Casos de Dengue', fontsize=12, fontweight='bold')
plt.xlabel('Google Trends (dengue)')
plt.ylabel('Casos')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/2_climate_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Análisis temporal avanzado
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Serie temporal de casos por año
ax1 = axes[0, 0]
for year in sorted(df['year'].unique()):
    year_data = df[df['year'] == year].groupby('week')['target_cases'].sum()
    ax1.plot(year_data.index, year_data.values, marker='o', label=str(year), linewidth=2, markersize=3)
ax1.set_title('Series Temporales por Año', fontsize=14, fontweight='bold')
ax1.set_xlabel('Semana del Año')
ax1.set_ylabel('Casos')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Boxplot por mes
ax2 = axes[0, 1]
df['month'] = df['week'].apply(lambda x: (x-1)//4 + 1)  # Aproximación semana a mes
monthly_data = [df[df['month'] == month]['target_cases'].values for month in range(1, 13)]
box_plot = ax2.boxplot(monthly_data, patch_artist=True)
colors = plt.cm.viridis(np.linspace(0, 1, 12))
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_title('Distribución de Casos por Mes', fontsize=14, fontweight='bold')
ax2.set_xlabel('Mes')
ax2.set_ylabel('Casos')
ax2.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
ax2.grid(True, alpha=0.3)

# Heatmap casos por semana y año
ax3 = axes[1, 0]
pivot_data = df.pivot_table(values='target_cases', index='week', columns='year', aggfunc='sum', fill_value=0)
sns.heatmap(pivot_data, ax=ax3, cmap='YlOrRd', cbar_kws={'label': 'Casos'})
ax3.set_title('Mapa de Calor: Casos por Semana y Año', fontsize=14, fontweight='bold')
ax3.set_xlabel('Año')
ax3.set_ylabel('Semana del Año')

# Evolución de Google Trends
ax4 = axes[1, 1]
trends_evolution = df.groupby(['year', 'week'])['trends_dengue'].mean().reset_index()
for year in sorted(trends_evolution['year'].unique()):
    year_trends = trends_evolution[trends_evolution['year'] == year]
    ax4.plot(year_trends['week'], year_trends['trends_dengue'], 
             marker='o', label=str(year), linewidth=2, markersize=3)
ax4.set_title('Evolución de Google Trends por Año', fontsize=14, fontweight='bold')
ax4.set_xlabel('Semana del Año')
ax4.set_ylabel('Google Trends (dengue)')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/3_temporal_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Análisis geográfico y demográfico
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Densidad poblacional vs casos
ax1 = axes[0, 0]
df['density'] = df['total_pobl'] / df['sup_km2']
df['density_log'] = np.log1p(df['density'])
ax1.scatter(df['density_log'], df['target_cases'], alpha=0.6, s=30)
z = np.polyfit(df['density_log'].dropna(), df[df['density_log'].notna()]['target_cases'], 1)
p = np.poly1d(z)
ax1.plot(df['density_log'].dropna().sort_values(), 
         p(df['density_log'].dropna().sort_values()), "r--", alpha=0.8, linewidth=2)
ax1.set_title('Densidad Poblacional vs Casos de Dengue', fontsize=14, fontweight='bold')
ax1.set_xlabel('Log(Densidad Poblacional + 1)')
ax1.set_ylabel('Casos')
ax1.grid(True, alpha=0.3)

# Distribución de superficie
ax2 = axes[0, 1]
ax2.hist(df['sup_km2'], bins=30, alpha=0.7, color='green', edgecolor='black')
ax2.axvline(df['sup_km2'].median(), color='red', linestyle='--', 
            label=f'Mediana: {df["sup_km2"].median():.1f} km²')
ax2.set_title('Distribución de Superficie por Departamento', fontsize=14, fontweight='bold')
ax2.set_xlabel('Superficie (km²)')
ax2.set_ylabel('Frecuencia')
ax2.legend()
ax2.grid(alpha=0.3)

# Casos por rango de población
ax3 = axes[1, 0]
df['pop_bins'] = pd.cut(df['total_pobl'], bins=5, labels=['Muy Baja', 'Baja', 'Media', 'Alta', 'Muy Alta'])
pop_cases = df.groupby('pop_bins', observed=True)['target_cases'].mean()
bars = ax3.bar(range(len(pop_cases)), pop_cases.values, 
               color=sns.color_palette("viridis", len(pop_cases)))
ax3.set_xticks(range(len(pop_cases)))
ax3.set_xticklabels(pop_cases.index, rotation=45)
ax3.set_title('Casos Promedio por Rango de Población', fontsize=14, fontweight='bold')
ax3.set_ylabel('Casos Promedio')
for i, v in enumerate(pop_cases.values):
    ax3.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')
ax3.grid(alpha=0.3)

# Top 15 departamentos por casos per cápita
ax4 = axes[1, 1]
df['cases_per_capita'] = (df['target_cases'] / df['total_pobl']) * 100000
dept_per_capita = df.groupby('departament_name')['cases_per_capita'].mean().sort_values(ascending=False).head(15)
bars = ax4.barh(range(len(dept_per_capita)), dept_per_capita.values, 
                color=sns.color_palette("plasma", len(dept_per_capita)))
ax4.set_yticks(range(len(dept_per_capita)))
ax4.set_yticklabels(dept_per_capita.index, fontsize=9)
ax4.set_title('Top 15 Departamentos - Casos per Cápita', fontsize=14, fontweight='bold')
ax4.set_xlabel('Casos por 100,000 habitantes')
ax4.invert_yaxis()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/4_geographic_demographic.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Análisis de outliers y distribuciones
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Boxplot de casos por año
ax1 = axes[0, 0]
yearly_data = [df[df['year'] == year]['target_cases'].values for year in sorted(df['year'].unique())]
box_plot = ax1.boxplot(yearly_data, patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(yearly_data)))
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_title('Distribución de Casos por Año', fontsize=12, fontweight='bold')
ax1.set_xlabel('Año')
ax1.set_ylabel('Casos')
ax1.set_xticklabels(sorted(df['year'].unique()))
ax1.grid(True, alpha=0.3)

# Distribución de Google Trends
ax2 = axes[0, 1]
ax2.hist(df['trends_dengue'].dropna(), bins=30, alpha=0.7, color='purple', edgecolor='black')
ax2.axvline(df['trends_dengue'].mean(), color='red', linestyle='--', 
            label=f'Media: {df["trends_dengue"].mean():.1f}')
ax2.set_title('Distribución de Google Trends', fontsize=12, fontweight='bold')
ax2.set_xlabel('Google Trends (dengue)')
ax2.set_ylabel('Frecuencia')
ax2.legend()
ax2.grid(alpha=0.3)

# QQ plot para casos
ax3 = axes[0, 2]
from scipy import stats
stats.probplot(df['target_cases'], dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot - Casos de Dengue', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Outliers en temperatura
ax4 = axes[1, 0]
Q1 = df['temp_2weeks_avg'].quantile(0.25)
Q3 = df['temp_2weeks_avg'].quantile(0.75)
IQR = Q3 - Q1
outliers_temp = df[(df['temp_2weeks_avg'] < Q1 - 1.5*IQR) | 
                   (df['temp_2weeks_avg'] > Q3 + 1.5*IQR)]
ax4.scatter(df.index, df['temp_2weeks_avg'], alpha=0.6, s=20, label='Normal')
ax4.scatter(outliers_temp.index, outliers_temp['temp_2weeks_avg'], 
            color='red', s=30, label=f'Outliers ({len(outliers_temp)})')
ax4.set_title('Outliers en Temperatura', fontsize=12, fontweight='bold')
ax4.set_xlabel('Índice')
ax4.set_ylabel('Temperatura (°C)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Casos extremos
ax5 = axes[1, 1]
extreme_cases = df[df['target_cases'] > df['target_cases'].quantile(0.95)]
ax5.scatter(df['week'], df['target_cases'], alpha=0.6, s=20, label='Normal')
ax5.scatter(extreme_cases['week'], extreme_cases['target_cases'], 
            color='red', s=50, label=f'Casos Extremos ({len(extreme_cases)})')
ax5.set_title('Casos Extremos por Semana', fontsize=12, fontweight='bold')
ax5.set_xlabel('Semana del Año')
ax5.set_ylabel('Casos')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Distribución log de casos
ax6 = axes[1, 2]
log_cases = np.log1p(df['target_cases'])
ax6.hist(log_cases, bins=30, alpha=0.7, color='brown', edgecolor='black')
ax6.axvline(log_cases.mean(), color='red', linestyle='--', 
            label=f'Media: {log_cases.mean():.2f}')
ax6.set_title('Distribución Log(Casos + 1)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Log(Casos + 1)')
ax6.set_ylabel('Frecuencia')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots/5_outliers_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Resumen estadístico y tabla de insights
print("\n" + "="*80)
print("RESUMEN ESTADÍSTICO DEL DATASET")
print("="*80)

# Estadísticas generales
print(f"\n📊 ESTADÍSTICAS GENERALES:")
print(f"   • Total de registros: {len(df):,}")
print(f"   • Período de tiempo: {df['year'].min()}-{df['year'].max()}")
print(f"   • Número de departamentos: {df['departament_name'].nunique()}")
print(f"   • Total de casos de dengue: {df['target_cases'].sum():,}")
print(f"   • Promedio de casos por registro: {df['target_cases'].mean():.2f}")

# Top insights
print(f"\n🔍 INSIGHTS PRINCIPALES:")
top_year = yearly_cases.idxmax()
top_dept = df.groupby('departament_name')['target_cases'].sum().idxmax()
peak_week = weekly_cases.idxmax()

print(f"   • Año con más casos: {top_year} ({yearly_cases[top_year]:,} casos)")
print(f"   • Departamento más afectado: {top_dept}")
print(f"   • Semana pico del año: {peak_week}")
print(f"   • Correlación temperatura-casos: {df['temp_2weeks_avg'].corr(df['target_cases']):.3f}")
print(f"   • Correlación Google Trends-casos: {df['trends_dengue'].corr(df['target_cases']):.3f}")

# Datos faltantes
print(f"\n❗ DATOS FALTANTES:")
missing_data = df.isnull().sum()
for col in missing_data[missing_data > 0].index:
    print(f"   • {col}: {missing_data[col]} ({missing_data[col]/len(df)*100:.1f}%)")

print(f"\n✅ Gráficos guardados en la carpeta 'eda_plots/'")
print("="*80) 