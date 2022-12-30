#!/usr/bin/perl

use strict;

my $pdb = $ARGV[0];
my $rosetta_dir = $ARGV[1];

if ($pdb eq "") { die("usage: run_ros_score.pl pdb_list\n"); }

my $outfile = "$pdb.ros.scores.out";
#my $score_wts = "-score:weights ddg";
#my $score_wts = "-beta_nov16";
my $score_wts = "";

# clean up
system("rm $outfile");


my $ros_bin = "$rosetta_bin/main/source/bin/score.static.linuxgccrelease";
my $ros_cmd = "$ros_bin -in:ignore_unrecognized_res -run:ignore_zero_occupancy false -in:file:l $pdb $score_wts -overwrite -out:output -out:file:scorefile $outfile -database /home/fabian/Desktop/Uni/Masterthesis/other_repos/antibody_benchmark/rosetta/rosetta_bin_linux_2017.29.59598_bundle/main/database/";
print "executing: $ros_cmd\n";
system($ros_cmd);
