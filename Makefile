# Makefile for building the project

# Source directories and files from the main repository
MAIN_SOURCE_DIRS = app

# Source directories and files from the docs repository
DOCS_SOURCE_DIR = ../spfluo-docs

DATA_DIR = data

# Build directory
BUILD_DIR = build

# Output zip file
ZIP_FILE = spfluo-app.zip

.PHONY: build

build:
	mkdir -p $(BUILD_DIR)
	cp -r $(MAIN_SOURCE_DIRS) $(MAIN_SOURCE_FILES) $(BUILD_DIR)
	. $(DOCS_SOURCE_DIR)/.venv/bin/activate && $(MAKE) -C $(DOCS_SOURCE_DIR) clean
	. $(DOCS_SOURCE_DIR)/.venv/bin/activate && $(MAKE) -C $(DOCS_SOURCE_DIR) html
	. $(DOCS_SOURCE_DIR)/.venv/bin/activate && $(MAKE) -C $(DOCS_SOURCE_DIR) latexpdf
	mkdir -p $(BUILD_DIR)/docs
	cp -r $(DOCS_SOURCE_DIR)/build/html $(BUILD_DIR)/docs
	cp $(DOCS_SOURCE_DIR)/build/latex/spfluo-app.pdf $(BUILD_DIR)/docs
	@echo "Build completed in $(BUILD_DIR)/"

zip: clean build
	mkdir $(BUILD_DIR)/spfluo-app
	cp -r $(BUILD_DIR)/docs $(BUILD_DIR)/spfluo-app/docs
	cp -r $(BUILD_DIR)/app $(BUILD_DIR)/spfluo-app/app
	cp -r $(DATA_DIR) $(BUILD_DIR)/spfluo-app/data
	cd $(BUILD_DIR) && zip -9 -r $(ZIP_FILE) spfluo-app
	rm -r $(BUILD_DIR)/spfluo-app
	@echo "Build zipped to $(ZIP_FILE)"

clean:
	rm -r $(BUILD_DIR)
	rm -f $(ZIP_FILE)
	@echo "Cleaned build directory and zip file"