# Transforming DeepGSR

Transforming DeepGSR is a light-weight model for recognizing genomic regulatory signals, such as polyadenylation sites (PAS) and translation initiation sites (TIS). Our approach significantly reduces model size while maintaining competitive performance. Using trinucleotide tokenization and models with fewer than 2.2 million parameters, we train on PAS and TIS datasets from human, mouse, bovine, and fruit fly. The models achieve accuracies of up to 83.5% for PAS and 93.1% for TIS. Compared to alternative low-capacity models, the Transformer outperforms in precision and recall, offering a practical solution for bioinformatics applications in resource-limited environments.

Classify your sequences at https://dvgodoy.github.io/DeepGSR