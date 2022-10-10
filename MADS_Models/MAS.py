##  model with simle LSTM for sentence encoding (without tree lstm)

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from . import Constants
import pandas as pd


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
    def __init__(self, vocab_size, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word, num_atten_head):
        super(DocLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.max_num_para = max_num_para
        self.max_num_sent = max_num_sent
        self.max_num_word_body = max_num_word
        self.att_head = num_atten_head
        self.max_num_word_head = max_num_word
        self.mem_dim = mem_dim
        self.in_dim = in_dim

        self.conv_embed= (2*self.mem_dim)
        self.num_filter=10
        self.window_sizes = (1,2,3,4,5,6,7)

        self.convs = nn.ModuleList([
                                   nn.Conv2d(1, self.num_filter, [window_size,self.conv_embed], padding=(window_size - 1, 0))
                                   for window_size in self.window_sizes
        ])

        self.sentence_head_BILSTM = nn.LSTM(in_dim, mem_dim, 1,bidirectional=True)

        torch.manual_seed(0)
        self.sim_head_sent = nn.Linear((2 * self.mem_dim)*(2*self.mem_dim) , 1)


        if self.att_head == 1:
            self.attention_head1= SingleHead_Self_Attention(2*self.mem_dim)
            self.concnate_head_output= nn.Linear(((2*self.mem_dim)*self.att_head), (2*self.mem_dim))
        elif self.att_head == 8 :
            self.attention_head1= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head2= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head3= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head4= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head5= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head6= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head7= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head8= SingleHead_Self_Attention(2*self.mem_dim)
            self.concnate_head_output= nn.Linear(((2*self.mem_dim)*self.att_head), (2*self.mem_dim))
        else:
            self.attention_head1= SingleHead_Self_Attention(2*self.mem_dim)
            self.attention_head2= SingleHead_Self_Attention(2*self.mem_dim)
            self.concnate_head_output= nn.Linear(((2*self.mem_dim)*self.att_head), (2*self.mem_dim))
        self.multihead_to_feature_map= nn.Linear((max_num_sent*2*mem_dim),(2*mem_dim))
        self.sent_pad =   torch.zeros(1, 2* mem_dim)
        self.para_pad = torch.zeros(1, 2* mem_dim) #torch.Size([1, 1, 150])
        self.word_pad= torch.randn(1, in_dim)


    def forward(self, body):

        rsent = body['headline']['rsent']


        rinputs_list=[]

        for word in rsent:
            rinputs_list.append(self.emb(word).view(1, self.in_dim))

        if len(rsent) < self.max_num_word_body:
             rinputs_list += [ self.word_pad] * (self.max_num_word_body - len(rinputs_list))

        seq_head_input=torch.cat(rinputs_list[:self.max_num_word_body],0)

        rinputs=self.emb(rsent)
        head_o, (head_h,head_c) = self.sentence_head_BILSTM(seq_head_input.contiguous().view(self.max_num_word_body, 1, self.in_dim))
        head_hid_2d=head_h.view(2,100)
        h_left=head_hid_2d[0]
        h_right=head_hid_2d[1]
        Bi_Head_h=torch.cat((h_left,h_right),0)
        del rinputs_list

        body=body['body_list']
        count=0
        sent_hidden_list = []
        for p_id in body:
            count=count+1
            if count > self.max_num_para:   # condition for only two paragrphs
               break

            for s_id, sentence in enumerate(body[p_id]):
                lsent = sentence
                linputs = self.emb(lsent)

                linputs_list=[]
                for word in lsent:
                    linputs_list.append(self.emb(word).view(1, self.in_dim))

                if len(lsent) < self.max_num_word_body:
                    linputs_list += [ self.word_pad] * (self.max_num_word_body - len(linputs_list))

                seq_head_input=torch.cat(linputs_list[:self.max_num_word_body],0)

                body_sent_o,(body_sent_h,body_sent_c) = self.sentence_head_BILSTM(seq_head_input.contiguous().view(self.max_num_word_body, 1, self.in_dim))

                body_hid_2d=body_sent_h.view(2,100)
                body_sent_left=body_hid_2d[0]
                body_sent_right=body_hid_2d[1]
                Bi_body_sent_h=torch.cat((body_sent_left,body_sent_right),0)
                del linputs_list
                sent_hidden_list.append(Bi_body_sent_h.view(1,self.mem_dim*2))

        "Merge encodings of sentence into doccuments"
        sent_hidden_list += [ self.sent_pad] * (self.max_num_sent - len(sent_hidden_list))
        sent_enc= torch.cat((sent_hidden_list[:self.max_num_sent]),0)
        # print("sentence encoding shape",sent_enc.shape)

        if self.att_head == 1:
            att1_op, att1_w= self.attention_head1(sent_enc,1)
            head_concat= att1_op
        elif self.att_head == 8:
            att1_op, att1_w= self.attention_head1(sent_enc,1)
            att2_op, att2_w= self.attention_head2(sent_enc,1)
            att3_op, att3_w= self.attention_head3(sent_enc,1)
            att4_op, att4_w= self.attention_head4(sent_enc,1)
            att5_op, att5_w= self.attention_head5(sent_enc,1)
            att6_op, att6_w= self.attention_head6(sent_enc,1)
            att7_op, att7_w= self.attention_head7(sent_enc,1)
            att8_op, att8_w= self.attention_head8(sent_enc,1)
            head_concat= torch.cat((att1_op,att2_op,att3_op,att4_op,att5_op,att6_op,att7_op,att8_op),1)
        else:
            att1_op, att1_w= self.attention_head1(sent_enc,1)
            att2_op, att2_w= self.attention_head2(sent_enc,1)
            head_concat= torch.cat((att1_op,att2_op),1)
        high_multihead_output= self.concnate_head_output(head_concat)




        final_high_rep_vec= torch.flatten(high_multihead_output)

        high_sent_feature= self.multihead_to_feature_map(final_high_rep_vec)

        x_h=[]
        for conv in self.convs:
            x=conv(sent_enc.view(1,1,self.max_num_sent,200))
            x = torch.squeeze(x, -1)
            x = F.max_pool1d(x, x.size(2))
            x_h.append(x)

        high_sim_pooled = torch.cat(x_h[:len(self.window_sizes)],0)
        high_sim_pooled= high_sim_pooled.view(7,10)
        # print("pooling feature shape",high_sim_pooled.shape)
        del x_h






        return high_sent_feature, Bi_Head_h, high_sim_pooled




# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes,max_num_word_head):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_num_word_head= max_num_word_head


        self.wh = nn.Linear((8 * self.mem_dim+70) , self.hidden_dim)  # for only deep feature.
        self.wp = nn.Linear(self.hidden_dim, 2)

    def forward(self, body,head,conv):
        conv= conv.view(70,1)

        head= head.view(2*self.mem_dim,1)
        body=body.view(2*self.mem_dim,1)

        "Angle and difference"
        diff = torch.abs(torch.add(body, - head))
        ang = torch.mul(body, head)
        feature_vec=torch.cat((head,body,diff,ang,conv),0)
        out = torch.sigmoid(self.wh(torch.t(feature_vec))) # for model with only deep feature
        out =self.wp(out) # No softmax
        return out




# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim,  sparsity, freeze, num_classes, \
        max_num_para, max_num_sent, max_num_word,  num_filter, num_head):
        super(SimilarityTreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.doclstm = DocLSTM(vocab_size, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word, num_head)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes,  max_num_word)
    def forward(self, body):
        body, head, conv= self.doclstm(body)
        output = self.similarity(body,head,conv)
        return output
