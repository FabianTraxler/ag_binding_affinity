((python-mode . ((eval . (blacken-mode -1)))))

(dap-register-debug-template
 "Python :: Run ABAG affinity training"
 (list :type "python"
       :args " -t model_train -b 2 -e 1 -n residue --target_dataset abag_affinity_of_embeddings:absolute --validation_size 1 --gnn_type ipa --aggregation_method edge --attention_heads 12"
       :cwd nil
       :module nil
       :program "/home/moritz/Projects/guided-protein-diffusion/modules/ag_binding_affinity/src/abag_affinity/main.py"
       :request "launch"
       :name "Python :: Run ABAG affinity training"))

;; TODO program is superseeded by target-module and we could enable '--relaxed_pdbs' at some point

