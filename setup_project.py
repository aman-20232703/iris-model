"""
Automated Project Setup Script
Run this FIRST to create the complete project structure
Usage: python setup_project.py
"""

import os
import sys

def create_directory_structure():
    """Create all necessary directories"""
    
    directories = [
        'data',
        'notebooks',
        'src',
        'models',
        'visualizations',
        'docs',
        'tests'
    ]
    
    print("="*60)
    print("IRIS CLASSIFICATION PROJECT SETUP")
    print("="*60)
    print("\nCreating directory structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    print("\n✓ Directory structure created successfully!")

def create_requirements_txt():
    """Create requirements.txt file"""
    
    requirements = """numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
pytest>=7.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("\n✓ Created: requirements.txt")

def create_gitignore():
    """Create .gitignore file"""
    
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
models/*.pkl
models/*.joblib
visualizations/*.png
data/*.csv
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore)
    
    print("✓ Created: .gitignore")

def print_next_steps():
    """Print clear instructions"""
    
    print("\n" + "="*60)
    print("SETUP COMPLETE! 🎉")
    print("="*60)
    
    print("""
Next Steps:
-----------

1. INSTALL DEPENDENCIES
   pip install -r requirements.txt

2. ALL SOURCE FILES ARE IN THE CHAT ARTIFACTS
   I will now provide each file one by one.
   Save them to the locations shown.

3. AFTER COPYING ALL FILES, RUN:
   python src/iris_classification.py

Your project structure is ready!
    """)

def main():
    """Main setup function"""
    
    try:
        create_directory_structure()
        create_requirements_txt()
        create_gitignore()
        print_next_steps()
        
        print("\n" + "="*60)
        print("Setup completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()