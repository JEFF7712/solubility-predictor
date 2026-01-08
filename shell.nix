{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {

  packages = [
    (pkgs.python3.withPackages (p: with p; [
      numpy
      pandas
      openpyxl
      scikit-learn
      matplotlib
      rdkit
      torch
      torch-geometric
    ]))
  ];
}
