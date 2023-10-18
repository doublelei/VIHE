conda create --name vihe python=3.8
conda activate vihe

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz -C VIHE/lib/ && rm 1.10.0.tar.gz
export CUB_HOME=$(pwd)/VIHE/lib/cub-1.10.0

conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python

# Tests/Linting
pip install black usort flake8 flake8-bugbear flake8-comprehensions
FORCE_CUDA=1 pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

# Get the Ubuntu release number
UBUNTU_VERSION=$(lsb_release -rs)

# Choose the correct download URL based on the version
case $UBUNTU_VERSION in
    16.04)
        FILENAME="CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz"
        ;;
    18.04)
        FILENAME="CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz"
        ;;
    20.04)
        FILENAME="CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
        ;;
    *)
        echo "Unsupported Ubuntu version: $UBUNTU_VERSION"
        exit 1
        ;;
esac

# # Download the file
# wget "https://www.coppeliarobotics.com/files/$FILENAME" --no-check-certificate

# # Extract the file
# tar -xf $FILENAME -C VIHE/lib/ && rm $FILENAME

# # Assuming the extracted folder is named the same as the tarball without the extension
DIRNAME=VIHE/lib/${FILENAME%.tar.xz}
echo $DIRNAME

# # Add the paths to the ~/.bashrc
echo "export COPPELIASIM_ROOT=$PWD/$DIRNAME" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT" >> ~/.bashrc
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> ~/.bashrc
echo "export DISPLAY=:1.0" >> ~/.bashrc

# # Feedback to the user
# echo "Paths added to ~/.bashrc. Please restart your terminal or run 'source ~/.bashrc'."

# git clone --recurse-submodules https://github.com/doublelei/VIHE.git && cd VIHE && git submodule update --init

pip install -e .
pip install -e VIHE/lib/PyRep 
pip install -e VIHE/lib/RLbench 
pip install -e VIHE/lib/YARR 