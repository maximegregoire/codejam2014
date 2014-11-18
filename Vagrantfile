# -*- mode: ruby -*-
# vi: set ft=ruby :

VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
 config.vm.box = "C:/windows7pro32.box"
 config.vm.communicator = "winrm"
 config.winrm.username = "vagrant"
 config.winrm.password = "vagrant"
 config.vm.network "forwarded_port", host: 33389, guest: 3389
end