import io
import os
import sys 
import argparse
import time
import torch
from Bio.Data import SCOPData
from Bio.PDB import PDBParser, PPBuilder
#from tape import TAPETokenizer, ProteinBertModel
from chimerax.core.commands import run
#from transformers import AutoModel, AutoTokenizer
from transformer_infrastructure.hf_embed import load_model
from transformer_infrastructure.attn_calc import get_attn_data, parse_mut


# When running through ChimeraX, put whole python command + args in quotes
# ex. python  /home/cmcwhite/Downloads/chimerax-1.3/lib/python3.9/site-packages/ChimeraX_main.py --nogui --script "/scratch/gpfs/cmcwhite/attentionview_provis/scripts/chart_attentions.py  -st /scratch/gpfs/cmcwhite/attentionview_provis/6vx6.pdb -mo /scratch/gpfs/cmcwhite/prot_bert_bfd -ma 0.1"

def get_attention_args():
    parser = argparse.ArgumentParser()
    #                    help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-mo", "--model", dest = "model_path", type = str, required = True,
                        help="Model directory Ex. /path/to/model_dir")
    parser.add_argument("-st", "--struct_file", dest = "struct_file", type = str, required = False,
                        help="A protein structure file, Optional")
    parser.add_argument("-p", "--pdb_id", dest = "pdb_id", type = str, required = False,
                        help="PDB identifier (ex. 6VX6), Optional")
   
    parser.add_argument("-ma", "--min_attn", dest = "min_attn", type = float, required = False, default= 0.1,
                        help="Minimum attention to plot, default: 0.1")
    parser.add_argument("-as", "--attn_scale", dest = "attn_scale", type = float, required = False, default= 0.9,
                        help="Amount to scale plotted cylinder widths, default: 0.9")
    parser.add_argument("-mu", "--mut", dest = "mut", type = str, required = False,
                        help="Mutate position X to Y. ex. '102_W' or 'P_102_W'. Won't change structure, just attention calculation ")

 
    args = parser.parse_args()
    print(args) 
    return(args)



def get_structure(pdb_id):
    resource = urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb_id}.pdb')
    content = resource.read().decode('utf8')
    handle = io.StringIO(content)
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, handle)

def get_tokens_and_coords(chain, mut =  None):

   if mut is not None:
       old, new, mutpos = parse_mut(mut)
   else:
       old = None
       new = None  
       mutpos = None
   #print(old, new, mutpos)
   coords = []
   tokens = [] 
   resnums = []
   for res in chain:
       resnum = res.id[1]
       coord = res['CA'].coord.tolist()
       token = SCOPData.protein_letters_3to1.get(res.get_resname(), "X") 
       #print(resnum, mutpos, token)
       if resnum == mutpos:
           print("replaced {} with {} at {}".format(token, new, mutpos))
           token = new      
       coords.append(coord)
       tokens.append(token)
       resnums.append(resnum) # Amino acid position, indexed at 1
   #print(len(tokens), len(coords), len(resnums))
   return(tokens, coords, resnums)
 

#pdb_id = '7HVP'


#model_rcsb_name = (model_pdb.id_string)



#def load_model(model_path):
#    '''
#    Takes path to huggingface model directory
#    Returns the model and the tokenizer
#    '''
#    print("load tokenizer")
#    tokenizer = AutoTokenizer.from_pretrained(model_path)
#
#    print("load model")
#    model = AutoModel.from_pretrained(model_path, output_attentions=True)

#    return(model, tokenizer)

def main_func():

    #args = sys.argv
    
    #print(sys.argv)
    args = get_attention_args()
    print(args)
  
    chain_ids = None # All chains

    struct_file = args.struct_file
    pdb_id = args.pdb_id
    model_path = args.model_path
    min_attn = args.min_attn
    attn_scale = args.attn_scale
    mut = args.mut

    if not pdb_id:
      if not struct_file:
         print("Requires either --pdb_id or --struct_file")
         return(0)

    model, tokenizer, model_config = load_model(model_path,
                        output_hidden_states=False,
                        output_attentions = True,
                        return_config = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available?", torch.cuda.is_available())
    model.to(device)
    # Set to inference mode
    model.eval()
 
    base_structfile = struct_file.split("/")[-1]
    if struct_file:
        parser = PDBParser(QUIET=True)
        structure =  parser.get_structure(base_structfile, struct_file)
    if pdb_id:
        structure = get_structure(pdb_id)

    

    structure_models = list(structure.get_models())
    if len(structure_models) > 1:
       print('Warning:', len(structure_models), 'models. Using first one')
    prot_model = structure_models[0]

    # Get tokens, coords, resnums for each chain
    # Get attentions between by concatenating. all tokens

    if chain_ids is None:
       chain_ids = [chain.id for chain in prot_model]

    tokens = []
    coords = []
    resnums = []

    for chain_id in chain_ids: 
        #chain_id = chain_ids[0]
    
        # Get informations out of structure file
        print('Loading chain', chain_id) 
        chain = prot_model[chain_id]   
        print("chain", chain) 
        chain_tokens, chain_coords, chain_resnums = get_tokens_and_coords(chain, mut)
        tokens = tokens + chain_tokens
        coords = coords + chain_coords
        resnums = resnums + chain_resnums
  


        print(tokens)
        # Calculate attentions 
    model.to(device)
    attns = get_attn_data(model, tokenizer, tokens,min_attn  = min_attn)
     
         

    num_layers = len(attns)
    num_heads = len(attns[0])

    if not os.path.isdir("bilds"):
        os.mkdir("bilds")


    run(session, f"open {struct_file}")
    if mut:
       newname = "{}-{}".format(mut, base_structfile)
       run(session, f"rename #1 {newname}")
    bild_filelist = []
    if mut:
        attn_outfile = "{}-{}-attns.csv".format(struct_file, mut)
    else:
        attn_outfile = "{}-attns.csv".format(struct_file)
    with open(attn_outfile, "w") as o:
       o.write("identifier,layer,head,res1,res2,attention\n")
       for layer in range(1, num_layers + 1): # Max 31
          print("layer {}".format(layer))
          for head in range(1, num_heads + 1): # Max 17
             
             identifier = "{}-{}-{}".format(base_structfile, layer, head)
             attn_head = attns[ layer - 1 ][ head - 1 ]
             #print(len(attn_head))
             #print(len(attn_head[0]))
             if mut: 
                bild_outfile = "bilds/{}-layer{}-head{}-{}.bild".format(base_structfile, layer, head, mut)
             else:
                bild_outfile = "bilds/{}-layer{}-head{}.bild".format(base_structfile, layer, head)
             bild_filelist.append(bild_outfile)
  
             with open(bild_outfile, "w") as bildfile:
                 bildfile.write(".color grey50\n")
                 bildfile.write(".transparency 90\n")

                 complete = []
                 for i in range(len(tokens)):
                    if i in complete:
                       continue
                    complete.append(i)
                    coords_i = " ".join([str(x) for x in coords[i]])
                    for j in range(len(tokens)):
                        coords_j = " ".join([str(x) for x in coords[j]])
                        attn = max(attn_head[i][j], attn_head[j][i])
                        if attn is not None and attn >= min_attn:

                            o.write("{},{},{},{}-{},{}-{},{}\n".format(identifier, layer, head, tokens[i], resnums[i], tokens[j], resnums[j], attn))
                            bildfile.write(".cylinder {} {} {}\n".format(coords_i, coords_j, attn))


              

    for bild in bild_filelist:
       run(session, f"open {bild}")
    run(session, "info models")
    if mut:
        outfile = "{}-{}-attns.cxs".format(struct_file, mut)
    else:
        outfile = "{}-attns.cxs".format(struct_file)
    run(session, f"save {outfile} format session compress gzip") # models #1,2")
    run(session, f"exit")
    return(0)
main_func()

