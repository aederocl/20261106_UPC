# Gaussian integral benchmark: make all | bench | plots | clean
BUILD=build
all: bench plots

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/integral_c: src/integral.c | $(BUILD)
	gcc -O2 $< -o $@ -lm
$(BUILD)/integral_f: src/integral.f90 | $(BUILD)
	gfortran -O2 $< -o $@
$(BUILD)/plot_c: src/plot.c | $(BUILD)
	gcc -O2 $< -o $@ -lm
$(BUILD)/plot_f: src/plot.f90 | $(BUILD)
	gfortran -O2 $< -o $@

bench: $(BUILD)/integral_c $(BUILD)/integral_f
	./$(BUILD)/integral_c
	./$(BUILD)/integral_f
	julia -O3 src/integral.jl
	python3 src/integral_np.py
	python3 src/integral.py        # slow: ~30 s

plots: $(BUILD)/plot_c $(BUILD)/plot_f
	cd figures && ../$(BUILD)/plot_c
	cd figures && ../$(BUILD)/plot_f
	cd figures && python3 ../src/plot.py
	cd figures && julia ../src/plot.jl

clean:
	rm -rf $(BUILD) figures/*.dat figures/*.gp
