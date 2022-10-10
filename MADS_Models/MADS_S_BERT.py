##  model with simle LSTM for sentence encoding (without tree lstm)

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from . import Constants

class SingleHead_Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(SingleHead_Self_Attention, self).__init__()
        self.input_dim= in_dim
        self.query_projection= nn.Linear(in_dim,in_dim)
        self.key_projection= nn.Linear(in_dim,in_dim)
        self.value_projection= nn.Linear(in_dim,in_dim)


    def forward(self, sent_mat, indicator):
        Query = self.query_projection(sent_mat)
        Key = self.key_projection(sent_mat)
        valaue = self.value_projection(sent_mat)
        if indicator == 1:
            similalrity= torch.matmul(Query, torch.t(Key))
        elif indicator == 0:
            similalrity= torch.matmul(Query, torch.t(Key))
            similalrity= torch.sub(1,similalrity)
        sim_norm= torch.div(similalrity,(torch.sqrt(torch.tensor(self.input_dim))))
        attention_weight= torch.softmax(sim_norm,dim=0)
        attention_output= torch.matmul(attention_weight,valaue)

        return attention_output, attention_weight





class DocLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word, num_atten_head, beta):
        super(DocLSTM, self).__init__()

        self.max_num_para = max_num_para
        self.max_num_sent = max_num_sent
        self.max_num_word = max_num_word
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.thresod = beta
        self.conv_embed= self.in_dim
        self.att_head =  num_atten_head
        self.num_filter=10
        self.window_sizes = (1,2,3,4,5,6,7)
        self.num_atten_head = num_atten_head
        self.convs = nn.ModuleList([
                                   nn.Conv2d(1, self.num_filter, [window_size,self.conv_embed], padding=(window_size - 1, 0))
                                   for window_size in self.window_sizes
        ])

        self.sentence_BILSTM = nn.LSTM(in_dim, mem_dim, 1,bidirectional=True)
        torch.manual_seed(0)
        self.sim_head_sent = nn.Linear((self.in_dim*self.in_dim) , 1)
        if self.att_head == 1:
            self.attention_head1= SingleHead_Self_Attention(self.mem_dim)
            self.concnate_head_output= nn.Linear(((2*self.mem_dim)*self.att_head), (2*self.mem_dim))
        elif self.att_head == 8 :
            self.attention_head1= SingleHead_Self_Attention(self.mem_dim)
            self.attention_head2= SingleHead_Self_Attention(self.mem_dim)
            self.attention_head3= SingleHead_Self_Attention(self.mem_dim)
            self.attention_head4= SingleHead_Self_Attention(self.mem_dim)
            self.attention_head5= SingleHead_Self_Attention(self.mem_dim)
            self.attention_head6= SingleHead_Self_Attention(self.mem_dim)
            self.attention_head7= SingleHead_Self_Attention(self.mem_dim)
            self.attention_head8= SingleHead_Self_Attention(self.mem_dim)
            self.concnate_head_output= nn.Linear(((2*self.mem_dim)*self.att_head), (2*self.mem_dim))
        else:
            self.attention_head1= SingleHead_Self_Attention(self.mem_dim)
            self.attention_head2= SingleHead_Self_Attention(self.mem_dim)
            self.concnate_head_output= nn.Linear(((2*self.mem_dim)*self.att_head), (2*self.mem_dim))


        self.concnate_head_output= nn.Linear(((self.mem_dim)*num_atten_head), (self.mem_dim))

        self.multihead_to_feature_map= nn.Linear((max_num_sent*mem_dim),(mem_dim))

        self.sent_pad =   torch.zeros(1, mem_dim)
        self.para_pad = torch.zeros(1, mem_dim) #torch.Size([1, 1, 150])
        self.word_pad= torch.randn(1, in_dim)







    def forward(self, body):

        rsent = torch.tensor(body['headline']['rsent']).view(1,self.in_dim)

        body=body['body_list']
        count=0
        sent_hidden_list = []
        for p_id in body:
            count=count+1
            if count > self.max_num_para:   # condition for only two paragrphs
               break

            for s_id, sentence in enumerate(body[p_id]):
                lsent =torch.tensor(sentence).view(1,self.in_dim)
                sent_hidden_list.append(lsent)


        head_sent_similarity=[]
        for vec in sent_hidden_list:

            Head_sent_mul= torch.matmul(torch.t(rsent), vec)
            sim_input= torch.flatten(Head_sent_mul)
            sim_output=torch.sigmoid(self.sim_head_sent(sim_input))
            head_sent_similarity.append(sim_output)

        head_sent_sim_vec=torch.cat(head_sent_similarity,0)

        sent_prob=torch.softmax(head_sent_sim_vec,dim=0)
        sent_hidden_mat=torch.cat((sent_hidden_list[:len(sent_hidden_list)]),0)
        attend_sent_mat=torch.mul((sent_prob.view(1,len(head_sent_similarity))),torch.t(sent_hidden_mat))
        del head_sent_similarity


        "Dividing Sentence into two set."

        attend_sent_mat=torch.t(attend_sent_mat)
        high_sim_sent_hidden=[]
        low_sim_sent_hidden=[]
        for i in range(len(sent_hidden_list)):
            if head_sent_sim_vec[i] > self.thresod:
                high_sim_sent_hidden.append(attend_sent_mat[i].view(1,self.in_dim))
            else:
                low_sim_sent_hidden.append(attend_sent_mat[i].view(1,self.in_dim ))
        del sent_hidden_list




        "Detils  number of sentence in high and low"
        a_high=len(high_sim_sent_hidden)
        b_low=(len(low_sim_sent_hidden))

        "High similiarty Decoder Side"
        if len(high_sim_sent_hidden) == 0:
            value, index = torch.max(head_sent_sim_vec,dim=0)
            high_sim_sent_decoder = attend_sent_mat[index]
            high_sim_sent_decoder=high_sim_sent_decoder.view(1,self.in_dim)

        else:
            high_sim_sent_decoder= torch.cat((high_sim_sent_hidden[:len(high_sim_sent_hidden)]), 0)
        "Low similairty Decoder Side"
        if len(low_sim_sent_hidden) == 0:
            value, index = torch.min(head_sent_sim_vec,dim=0)
            low_sim_sent_decoder = attend_sent_mat[index]
            low_sim_sent_decoder= low_sim_sent_decoder.view(1,self.in_dim)

        else:
            low_sim_sent_decoder= torch.cat((low_sim_sent_hidden[:len(low_sim_sent_hidden)]), 0)

        "High related setences Convolutions"
        high_sim_sent_hidden += [ self.sent_pad] * (self.max_num_sent - len(high_sim_sent_hidden))
        high_sim_sent_conv= torch.cat(high_sim_sent_hidden[:self.max_num_sent], 0)

        x_h=[]
        for conv in self.convs:
            x=conv(high_sim_sent_conv.view(1,1,self.max_num_sent,self.in_dim))
            x = torch.squeeze(x, -1)
            x = F.max_pool1d(x, x.size(2))
            x_h.append(x)

        high_sim_pooled = torch.cat(x_h[:len(self.window_sizes)],0)
        high_sim_pooled= high_sim_pooled.view(7,10)
        del x_h

        "Low related setences Convolutions"

        low_sim_sent_hidden += [ self.sent_pad] * (self.max_num_sent - len(low_sim_sent_hidden))
        low_sim_sent_conv= torch.cat(low_sim_sent_hidden[:self.max_num_sent], 0)

        x_l=[]
        for conv in self.convs:
            x=conv(low_sim_sent_conv.view(1,1,self.max_num_sent,self.in_dim))

            x = torch.squeeze(x, -1)
            # print("the shape after convolutions",x.shape)
            x = F.max_pool1d(x, x.size(2))
            x_l.append(x)
        low_sim_pooled = torch.cat(x_l[:len(self.window_sizes)],0)
        low_sim_pooled= low_sim_pooled.view(7,10)
        del x_l

        "Final Convolutions output"
        low_sim_conv_vec= torch.flatten(high_sim_pooled)
        high_sim_conv_vec= torch.flatten(low_sim_pooled)
        "Final convolutions output end Here"

        "Summary decoder code."

        if self.att_head == 1:
            att1_op, att1_w= self.attention_head1(high_sim_sent_decoder,1)
            head_concat= att1_op
        elif self.att_head == 8:
            att1_op, att1_w= self.attention_head1(high_sim_sent_decoder,1)
            att2_op, att2_w= self.attention_head2(high_sim_sent_decoder,1)
            att3_op, att3_w= self.attention_head3(high_sim_sent_decoder,1)
            att4_op, att4_w= self.attention_head4(high_sim_sent_decoder,1)
            att5_op, att5_w= self.attention_head5(high_sim_sent_decoder,1)
            att6_op, att6_w= self.attention_head6(high_sim_sent_decoder,1)
            att7_op, att7_w= self.attention_head7(high_sim_sent_decoder,1)
            att8_op, att8_w= self.attention_head8(high_sim_sent_decoder,1)
            head_concat= torch.cat((att1_op,att2_op,att3_op,att4_op,att5_op,att6_op,att7_op,att8_op),1)
        else:
            att1_op, att1_w= self.attention_head1(high_sim_sent_decoder,1)
            att2_op, att2_w= self.attention_head2(high_sim_sent_decoder,1)
            head_concat= torch.cat((att1_op,att2_op),1)
        high_multihead_output= self.concnate_head_output(head_concat)

        number_of_sent_vector= high_multihead_output.shape[0]
        # print(number_of_sent_vector)

        if self.max_num_sent> number_of_sent_vector:
            sent_pad= torch.zeros((self.max_num_sent -number_of_sent_vector),self.mem_dim)
            final_high_rep_mat=torch.cat((high_multihead_output,sent_pad),dim=0)
        else:
            indices=[range(self.max_num_sent)]
            final_high_rep_mat= high_multihead_output[indices]

        final_high_rep_vec= torch.flatten(final_high_rep_mat)
        high_sent_feature= self.multihead_to_feature_map(final_high_rep_vec)
        #print(high_sent_feature.shape)



        if self.att_head == 1:
            att1_op, att1_w= self.attention_head1(low_sim_sent_decoder,0)
            head_concat= att1_op
        elif self.att_head == 8:
            att1_op, att1_w= self.attention_head1(low_sim_sent_decoder,0)
            att2_op, att2_w= self.attention_head2(low_sim_sent_decoder,0)
            att3_op, att3_w= self.attention_head3(low_sim_sent_decoder,0)
            att4_op, att4_w= self.attention_head4(low_sim_sent_decoder,0)
            att5_op, att5_w= self.attention_head5(low_sim_sent_decoder,0)
            att6_op, att6_w= self.attention_head6(low_sim_sent_decoder,0)
            att7_op, att7_w= self.attention_head7(low_sim_sent_decoder,0)
            att8_op, att8_w= self.attention_head8(low_sim_sent_decoder,0)
            head_concat= torch.cat((att1_op,att2_op,att3_op,att4_op,att5_op,att6_op,att7_op,att8_op),1)
        else:
            att1_op, att1_w= self.attention_head1(low_sim_sent_decoder,0)
            att2_op, att2_w= self.attention_head2(low_sim_sent_decoder,0)
            head_concat= torch.cat((att1_op,att2_op),1)
        low_multihead_output= self.concnate_head_output(head_concat)
        # print("final multihead output",low_multihead_output.shape)
        number_of_sent_vector= low_multihead_output.shape[0]
        # print(number_of_sent_vector)

        if self.max_num_sent> number_of_sent_vector:
            sent_pad= torch.zeros((self.max_num_sent -number_of_sent_vector),self.mem_dim)
            final_low_rep_mat=torch.cat((low_multihead_output,sent_pad),dim=0)
        else:
            indices=[range(self.max_num_sent)]
            final_low_rep_mat= low_multihead_output[indices]

        final_low_rep_vec= torch.flatten(final_low_rep_mat)
        low_sent_feature= self.multihead_to_feature_map(final_low_rep_vec)

        return high_sim_conv_vec, low_sim_conv_vec, rsent, low_sent_feature, high_sent_feature





# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self,in_dim, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear((6 * self.in_dim) , self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, 4)

    def forward(self, h_conv,  l_conv, head, low ,high):
        h_conv= h_conv.view(70,1)
        l_conv= l_conv.view(70,1)
        head= head.view(self.in_dim,1)
        low=low.view(self.in_dim,1)
        high=high.view(self.in_dim,1)
        "Convulations related features."
        conv_angle = torch.mul(h_conv, l_conv)
        conv_diff = torch.abs(torch.add(h_conv, - l_conv))
        conv_angle_diff = torch.cat((conv_angle, conv_diff),0 )
        conv_raw_cat = torch.cat((h_conv,l_conv),0)
        conv_feature=torch.cat((conv_angle_diff,conv_raw_cat),0)
        head_conv= torch.cat((head,conv_feature),0)

        "Summary related features"
        low_head_diff = torch.abs(torch.add(low, - head))
        high_head_diff = torch.abs(torch.add(high, - head))


        low_head_angle = torch.mul(low, head)
        high_head_angle = torch.mul(high, head)

        feature_vec=torch.cat((low_head_diff,high_head_diff,low_head_angle,high_head_angle,low,high),0)







        """ Merge the feature vecot befor going to MLP"""

        out = torch.sigmoid(self.wh(torch.t(feature_vec))) # for model with only deep feature
        out =self.wp(out) # No softmax
        return out




# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, hidden_dim, sparsity, freeze, num_classes, \
        max_num_para, max_num_sent, max_num_word, num_atten_head, beta):
        super(SimilarityTreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.doclstm = DocLSTM( in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word,num_atten_head, beta)
        self.similarity = Similarity(in_dim, mem_dim, hidden_dim, num_classes)
    def forward(self, body):
        high_conv, low_conv, head ,low_rep, high_rep = self.doclstm(body)
        output = self.similarity(high_conv,low_conv,head,low_rep, high_rep)
        return output
