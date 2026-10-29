"""
The model for birds-of-a-feather-resnet.

ResNet-18 from torchvision, adapted for 32×32 images (3×3 first convolution, no
max-pool), plus a 2-layer MLP projection head, trained with the NT-Xent contrastive
loss on pairs of augmented crops.

forward() accepts return_cache=True and returns a dict of internal activations, and a
patch argument (or forward hooks) overwrites a named activation for interventions.
"""

# TODO: implement. See README.md for the plan.
