cd ~

git clone https://github.com/fossi-foundation/open_pdks.git

cd open_pdks

./configure \
    --enable-sky130-pdk \
    --with-sky130-variants=A

make -j$(nproc)

sudo make install

echo 'export PDK_ROOT=/usr/local/share/pdk' >> ~/.bashrc

source ~/.bashrc

## Magic

sudo apt-get update
sudo apt-get install -y build-essential git m4 tcsh csh \
  libx11-dev tcl-dev tk-dev libcairo2-dev \
  mesa-common-dev libglu1-mesa-dev libncurses-dev

git clone https://github.com/rtimothyedwards/magic.git
cd magic
./configure
make
sudo make install

export PATH=/usr/local/bin:$PATH


echo 'export PDK_ROOT=/usr/local/share/pdk' >> ~/.bashrc

source ~/.bashrc
