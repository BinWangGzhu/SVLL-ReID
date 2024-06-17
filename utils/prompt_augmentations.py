import torch
# Here, all prompts are unique_prompts.
def swap(prompts,scale=0.3):
    aug1_prompts = prompts.clone()
    aug2_prompts = prompts.clone()
    # Get the shape of prompts.
    num_prompts, num_words, _ = prompts.shape
    # Calculate the number of word vectors that need to be swapped.
    num_to_swap = int(scale * num_words)
    # Process each prompt.
    for i in range(num_prompts):
        # Randomly select word vectors that need to be swapped.
        indices_to_swap_1 = torch.randperm(num_words)[:num_to_swap]
        indices_to_swap_2 = torch.randperm(num_words)[:num_to_swap]
        # Randomly generate new positions.
        new_positions_1 = torch.randperm(num_to_swap)
        new_positions_2 = torch.randperm(num_to_swap)
        # Swap positions.
        temp_1 = aug1_prompts[i, indices_to_swap_1].clone()
        temp_2 = aug2_prompts[i, indices_to_swap_2].clone()
        aug1_prompts[i, indices_to_swap_1] = temp_1[new_positions_1]
        aug2_prompts[i, indices_to_swap_2] = temp_2[new_positions_2]
    return aug1_prompts, aug2_prompts

def mask(prompts,scale=0.5):
    aug1_prompts = prompts.clone()
    aug2_prompts = prompts.clone()
    # Get the shape of prompts
    num_prompts, num_words,dim = prompts.shape
    # Process each prompt.
    for i in range(num_prompts):
        # Generate  masks
        mask_1 = torch.rand(num_words, dim) < scale
        mask_2 = torch.rand(num_words, dim) < scale
        ## Set the elements at corresponding positions to 0
        aug1_prompts[i][mask_1]=0
        aug2_prompts[i][mask_2]=0
    return aug1_prompts, aug2_prompts
def mix(prompts,scale=0.3):
    aug1_prompts = prompts.clone()
    aug2_prompts = prompts.clone()
    # Get the shape of prompts
    num_prompts, _, _= prompts.shape
    for i in range(num_prompts):
        # Randomly choose a prompt, but it cannot be the current prompt.
        aug1_randint = torch.randint(0, num_prompts - 1, (1,)).item()
        aug2_randint = torch.randint(0, num_prompts - 1, (1,)).item()
        # Calculate the new prompt.
        aug1_prompts[i] = (1 - scale) * aug1_prompts[i] + scale * aug1_prompts[aug1_randint]
        aug2_prompts[i] = (1 - scale) * aug2_prompts[i] + scale * aug2_prompts[aug2_randint]
    return aug1_prompts, aug2_prompts




def gausssiam_white(sentence):#
    snr=torch.randint(6,12,(sentence.shape[:2])).cuda()[:,:,None]
    P_signal=(sentence*sentence).sum(dim=-1,keepdim=True)/sentence.shape[-1]
    P_noise = P_signal / (10 ** (snr / 10.0))
    return sentence+torch.randn(sentence.shape).cuda()*torch.sqrt(P_noise)


def fft_ifft(sentence):#
    fft_a = torch.fft.fft(sentence-torch.mean(sentence))
    a_f = torch.fft.ifft(fft_a)
    return a_f.real

def add_noise(sentence):#
    dropout= torch.dropout(sentence,p=0.3,train=True)
    print(dropout)
    max_a,_=torch.max(sentence,dim=-1,keepdim=True)
    min_a,_=torch.min(sentence,dim=-1,keepdim=True)
    samples=torch.rand(sentence.shape)
    scale=(max_a*0.8-min_a)
    samples=samples*scale+min_a
    s_with_bg = sentence+samples*torch.Tensor(1).uniform_(0, 0.01)
    return s_with_bg


def wangqian(sentence,p1=0.3,p2=0.2,p3=0.2,p4=0.3):
    res1,res2,res3,res4=gausssiam_white(sentence),fft_ifft(sentence),add_noise(sentence),torch.dropout(sentence,p=0.3,train=True)
    mask=torch.rand(res1.shape[:2]).cuda()[:,:,None]
    mask1,mask2,mask3,mask4=(0<=mask)&(mask<p1),(p1<=mask)&(mask<p1+p2),(p1+p2<=mask)&(mask<p1+p2+p3),(p1+p2+p3<=mask)&(mask<=p1+p2+p3+p4)
    return  res1*mask1+res2*mask2+res3*mask3+res4*mask4

def select_aug(sentence,p1=0.6,p2=0.6):
    inputs_embeds=sentence
    s1,s2=wangqian(inputs_embeds),wangqian(inputs_embeds)# a sentence is inputed twice
    mask=torch.rand(s1.shape[:2]).cuda()[:,:,None]
    mask1=(0<=mask)&(mask<=p1)
    original=inputs_embeds*mask1
    s1=s1*(~mask1)
    s1=original+s1
    #
    mask=torch.rand(s2.shape[:2]).cuda()[:,:,None]
    mask2=(0<=mask)&(mask<=p2)
    original=inputs_embeds*mask2
    s2=s2*(~mask2)
    s2=original+s2
    return s1,s2


if __name__ == "__main__":
    tensor_2d = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    s1=add_noise(tensor_2d)
    print()