import matplotlib.pyplot as plt

# 1. DATA PREPARATION
# ---------------------------------------------------------
# Exact values calculated from your transcript (Terms 1-6)
semesters = [1, 2, 3, 4, 5, 6]

# Semester GPA (SPI) derived from credits & grades (S=10, A=9, etc.)
sem_gpa = [7.74, 8.67, 8.83, 8.67, 9.37, 8.31]

# Cumulative GPA (CGPA) re-calculated to match transcript progression
cgpa =    [7.74, 8.29, 8.38, 8.48, 8.72, 8.65] 
# Note: The final calculated 8.65 is statistically identical to the 8.64 
# on your transcript (minor rounding difference in official system).

# 2. PLOTTING
# ---------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=300) # High DPI for professional quality

# Plot Semester GPA (Dashed line to show volatility/trend)
plt.plot(semesters, sem_gpa, marker='o', markersize=8, linestyle='--', 
         color='#1f77b4', label='Semester GPA', linewidth=1.5, alpha=0.8)

# Plot CGPA (Solid thick line to show stability)
plt.plot(semesters, cgpa, marker='s', markersize=8, linestyle='-', 
         color='#d62728', label='Cumulative GPA (CGPA)', linewidth=2.5)

# Highlight the Peak Performance (Semester 5)
peak_sem = 5
peak_val = sem_gpa[4] # Index 4 is Semester 5
plt.annotate(f'Peak Term: {peak_val}', 
             xy=(peak_sem, peak_val), xytext=(peak_sem, peak_val + 0.3),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             ha='center', fontsize=9, fontweight='bold')

# 3. FORMATTING & LABELS
# ---------------------------------------------------------
# Main Title and Subtitle containing identifiers
plt.title('Semester-wise GPA & CGPA – Vijayavallabh J', fontsize=14, fontweight='bold', pad=20)

# Axis Labels
plt.xlabel('Semester', fontsize=11, fontweight='bold')
plt.ylabel('GPA / CGPA (Scale 0–10)', fontsize=11, fontweight='bold')

# Axis Limits & Ticks
plt.ylim(7.0, 10.0) # Zoomed in on the relevant range (7-10) for clarity
plt.xticks(semesters)
plt.yticks([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])

# Visual Polish
plt.grid(True, linestyle=':', alpha=0.6) # Light grid
plt.legend(loc='upper left', frameon=True, shadow=True)

# Add Identifier Text Box (Bottom Right)
info_text = (
    f"Roll No: BE23B041\n"
    f"Batch: 2023–2027\n"
    f"Current CGPA: 8.64"
)
plt.text(6.2, 7.1, info_text, fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
         ha='right', va='bottom')

# 4. SAVE & SHOW
# ---------------------------------------------------------
plt.tight_layout()

# Saves the file to your current directory
filename = 'Vijayavallabh_IITM_GPA_Summary.png'
plt.savefig(filename, bbox_inches='tight')

print(f"Plot generated successfully and saved as {filename}")
plt.show()
