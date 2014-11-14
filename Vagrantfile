# -*- mode: ruby -*-
# vi: set ft=ruby :

VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
 config.vm.box = "C:/windows7pro32.box"
 config.vm.communicator = "winrm"
 config.winrm.username = "vagrant"
 config.winrm.password = "vagrant"
 config.vm.network "forwarded_port", host: 33389, guest: 3389
 config.vm.provision :shell, :inline => "C:/vagrant/NDP451-KB2858728-x86-x64-AllOS-ENU.exe /q /norestart"
end