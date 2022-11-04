#!/bin/bash
#SBATCH --job-name=explain_preds        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=256G         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate hf-transformers


nvidia-smi

python -m torch.utils.collect_env

#python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/apolipo_social_model/ -o /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.train.fulllength.explain  -f /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo.fasta -l /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTrain.csv  


#python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/apolipo_social_model/ -o /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.test.fulllength.explain  -f /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo.fasta -l /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTest.csv 


python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/apolipo_social_model/ -o /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.AVIR.explain -a /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.AVIR.explain.attrib  -s MLGKCTTGCHGIHPKGYQEGHTYVYQLTGKTVTSVSQAHEEANLDLSATVELSVKPDCMHQLWLKDVLINNERTQLPGLDSAIQFNYHNGHIDTELCASPDDPQASLNIKRAVISLFQSAVLQDSGTATHHETDVLGVCPTSFSFNKKGDILSVEKTRNLEQCSFRENSNQGLISTTVDTAAGLKTSPLLASSQTIKQRFKQSVLESATSTEVHKLRPFSNGESGAKTVVETTLVLKSEKPDNPTAPVTLPKSLIFEVPHPVVKYQSKDIANAVKAAQEEIAGGVKPEAASKFADLVKVLRQSSSKDILSIYQKVKAGAGFDKASQQVFLDGLFHAGNGEAVEIGVKLLKNKELTDVQTLFFYTSLALIRHAQLPTVQAVTTLLDQPDLPRIGYLGIGQVIGKYCQDHPCENVPEVKEAVHKIREKVGNGKAKTREQENTVVSALKSLGNTRFLDDATLTKLANIASDKTVKNRVRVAALETLPTRCSMKWKNIAFKLLADKEQDSEIRIKTYLALVACPCEHVAGQLKEILDKETVNQVGSFIQTHLRNLRASTDPGKTNAKIYLGQIKPRTKFPEDVRKFSYNDELSYKIDSLGLGSTVEENVIYSQDSFVPRSANLNVTIQLFGRNLNLLDLSARVENLDRVIEHWFGPKGELWNHDLEDVKKEVKDAKETASSIGNYAKKKLDQLVRGKRAAVDQHELDRFAKKVHLRSNEIDQNLDLDVSLKLFGVELAYLSWQDESVGGKHPENAIIDKIFEKLQKAFDAAKNLHYDAENYVQFLEAELVYPTNLGTSLSLGLTGTSAVRVHTHTNVDVAKILQEPKNAAVKIALEPSVSVRIAGNIVVEALGTEAGMKVVSTLHTATSTDLTVNVLNGKGVDVQFSIPKKKQEVINVNSEVLVGNKGKYKPANFGKTNDFRDCFGQLAPALGLTVCGEVSFPYTDVATVQQKALFPLSGPAKFAVTVENVDANSYHFKVHLNDEEPGKRSFEILFETPGSKTKREISLTGEIGTQPDLHADLRLTSPIKNIQVGAALKNNEKERTLTVQLVHENVKYYGRVGIEVNGNKYKPVLDYNIPEHIEKLAGVNVAGKSGKKYSVQGTVDVAEDGKKYTTDKIALLMDDQEIVSISGSAGWQPNLLNIDAKIGYAGKQLALTVSAAGVLGTSYSLEVVAKPSSDPNLAFELFWELLNVEGSLDHKLTFVHGPDFKSETNRFSLKQNAVYKLKPNNLHLSTHNELKYPAVNLKVKLNGRLTQKSVDAEAEFKYEKFNVGGKLSSTVDKEKPGDYELELEAELLKNKIELKTKSAVVDPSTRTYVTSLEISPGGKYDSDVKLTWNMNKGDAVVGLNADANLNGKPITVTVNLDANKQLLNALASVKVNGAKQIEGTIKSTTTPHPTGHLNLFIKNYLTATGQYSYKAGNGQANLDVDIPTINRKIQATGHAVINQPEYSGTLEILYDAKKDPNKRVKLTTNNHITKTAVDSKNVIEAFEKKLEVNGKHSLQGDWNNGVLEQNLDVTLPDGQNINGKLNGKMVKKDNVKDFVVDSELAYSQKKGGPSTKLLLKDNGKFVSGKEPEIQRSVALKLVDTEGKDMGAVLNVGKKGQDSKMKESLDLEVSGAKIPKTVKLTGQYDKDEQGRQIDVEGSWGDDWKVSVRRLYDKSKLKHQLTVEVDLKKFNVPAEFGLKTNTQYNYPQLDHSSSLYLHSTKEKSEYTYQLYIHPKESASILTLPTREVALIVQHDLPKTKQTGAYKLDVSLYLDKKNKPSDKTSLTANGDINIDKNSLSLSGDAKFSYPTQTKDLSVKGRLHCGSEKLLDANLDLDVFDKKSQKITITANVQRQPVSDGYNITSNIDVNSRGQQLKLNEKSTFVLTTGDIELSSSFSYNDIHQKTKTLEFHFSADKNHALLLIALPDKILLKDDWTFQLSKESQSLNGELTILGEPTEVLKIEGKGLSSFELTNFRKDNPNNKLTLNGQLVLGQLAEIHADTYKDGAKKNLFSILLHLDEKQFLKPDISYDKANVNEVLIESNNFRLNLLKKLKETQEEVVKEVTVETGDLFKHLEKALPTLKPLAEYYKKELNKLKEEINADETVKEIQATLILVNNKRADYPANTENLHGYLVLPFTNVKSDYGVRISCTHKAPLVCAVHVSGFYHGKLRGLLGDGNNEPHDDFTLPSGTX --maxlength 250 

#python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/apolipo_social_model/ -o /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.train.500.explain  -f /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo.fasta -l /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTrain.csv   --maxlength 500


#python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/apolipo_social_model/ -o /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.test.fulllength.explain  -f /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo.fasta -l /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTest.csv  --maxlength 500



#python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/apolipo_social_model/ -o /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.train.1000.explain  -f /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo.fasta -l /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTrain.csv   --maxlength 1000


#python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/apolipo_social_model/ -o /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.test.fulllength.explain  -f /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo.fasta -l /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTest.csv  --maxlength 1000


#python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/apolipo_social_model/ -o /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.train.explain  -f /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo.fasta -l /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTrain.csv --maxlength 250 


#python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/apolipo_social_model/ -o /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_social_model.test.explain  -f /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo.fasta -l /scratch/gpfs/cmcwhite/apolipo_social_model/apolipo_labeledsetTest.csv --maxlength 250



#python transformer_infrastructure/hf_interpret.py -m /scratch/gpfs/cmcwhite/chloro/ -s MAMMMRKAAAVPASSRRSVAVNSVSGKRTVSGKAGAPVPEDVLAYAKTLPGVTAPFDNVFDPAGFLATASVKDVRRWRESEITHGRVAMLAALGFIVGEQLQDFPLFFNFDGRVSGPAIHFQQIGQGFWEPLLIAIGVAESYRVAVGWATPTGTGFNSLKDDYEPGDLGFDPLGLKPTDPEELKTLQTKELNNGRLAMIAIAAFVAQELVEQTEIFEHLVLRFEKEVILELEDVERDLGLPLTPLPDNLKAI -o lhsr1_chloro.explain -a lhsr1_chloro.defattr



