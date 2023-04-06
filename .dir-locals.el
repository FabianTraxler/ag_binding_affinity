((python-mode . ((eval . (blacken-mode -1)))))

(dap-register-debug-template
 "Python :: Run ABAG affinity training"
 (list :type "python"
       :args " -t model_train -b 2 -e 2 -n residue --target_dataset abag_affinity_of_embeddings:absolute --validation_size 1 --gnn_type identity --pretrained_model IPA --aggregation_method interface_sum"
       :cwd nil
       :module nil
       :program "/home/moritz/Projects/guided-protein-diffusion/modules/ag_binding_affinity/src/abag_affinity/main.py"
       :request "launch"
       :name "Python :: Run ABAG affinity training"))

;; TODO program is superseeded by target-module and we could enable '--relaxed_pdbs' at some point

