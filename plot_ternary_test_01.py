import ternary

# Boundary and Gridlines
scale = 500
figure, tax = ternary.figure(scale=10)

# Draw Boundary and Gridlines
tax.boundary(linewidth=1.0)
tax.gridlines(color="black", multiple=2)
# tax.gridlines(color="blue", multiple=1, linewidth=0.5)

# Set Axis labels and Title
fontsize = 12
tax.set_title("Simplex Boundary and Gridlines", fontsize=fontsize)
tax.left_axis_label("T2*", fontsize=fontsize)
tax.right_axis_label("PD", fontsize=fontsize)
tax.bottom_axis_label("T1w", fontsize=fontsize)

# Set ticks
tax.ticks(axis='lbr', linewidth=1)

# Remove default Matplotlib Axes
tax.clear_matplotlib_ticks()

figure.show()
