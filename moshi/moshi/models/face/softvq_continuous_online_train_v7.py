#!/usr/bin/env python
"""Locked v7.2 UniLS paper-training entry point.

This is the v7.2 counterpart of ``softvq_continuous_online_train_v7_e2e.py``:
the complete paper configuration is embedded below and command-line overrides
are rejected except for per-rank ``--batch-size``.  The locked values are the
ones stored in the evaluated checkpoint:

  outputs/SoftVQ_v7_2_unils_partner_no_neck_acc/checkpoint_epoch_125.ckpt

In particular, this is the successful no-neck-acceleration run
(``v72_neck_acc_weight=0``), not the obsolete launcher default of 2.0.

When moving servers, edit only PORTABLE PATH CONFIG.  Do not use environment
variables to alter v7.2 paper hyperparameters; this entry point overwrites them
with the locked values before importing the training entry point.

Runtime integration contract
----------------------------
* Own and partner Mimi features are causal 12.5-Hz inputs.
* Mimi semantic level-0 tokens are required for MTP.  ``*_token.pt`` files are
  only the offline cache of tokens supplied directly by Mimi at deployment.
* ``llm_feat`` remains the reserved 4096-D dialogue-LLM path.  This file does
  not repurpose or synthesize that input.
* VAP is disabled.  Partner conditioning, causal role conditioning, partner
  EMA/dropout, scheduled sampling, and excess quiet-jaw acceleration are on.

This file embeds exact checksum-verified copies of the v6, v7, v7.1, and v7.2
trainer lineage.  It therefore needs no sibling trainer Python files at runtime;
only the project core package and the configured data/checkpoint assets remain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


# BEGIN GENERATED STANDALONE TRAINER PAYLOAD
import base64
import hashlib
import types
import zlib

_EMBEDDED_TRAINER_PAYLOADS = {
    "softvq_continuous_online_train_v6": "c-rl~`*+($mLUAQ{t66prUS+#B`UU0Mh<gEvE`08evPH1zgZtG1d*VGh$I*S)WaV4-@f-%PZU7PcK6Qi`TESnRH5qD`__HmZ~kR>U9NV|^TlqqxQ>>0)n&2R>2x~RkE3dp=8J4~5MA#-IG^Q<t0=yH7{QM$y~w)J+n2u`N2@FySNV0;pXFthE%L=hG%gmCyvhswW3aPxIGL1DbyGy+beW##Gx(I1QL%`sVi^_FsJhG|0J~fjll2&zMe|~k&359mv$B|0*MBAeu*w(fVqGT10=rLWWXbjZV0m|T7Tx64C4u2VX4l!GiY~Ks5<S?-?y^}v>8HzVRjpT9G%Lz7>i45E%c4m!E_bi9RSBIYmoWZfbq7F`sz|OM59X8Z!Ol(;4F}O7jD20Ev*_*NyOY;P@A`+oJ$vyc$`)f7=_)EKScUl@ie4;mJ!a8ly_{vY(Ii{Win}?E21W;<;D7m|%2u;tyoS!cEu(Zj$%`m2VO6iQUQ`xP6aIq#OQTtK0c$l+s|;FAvlVog^`SsRTCS@oUc-u9Wzp}4Z=>^UR@`)P92GRK5UARdc|j{ny9Em$-CU;Dpx&JPa`Y}LmsxrR(}B**^bT9)3uq6^2)hIee+P?JRWtl>ou{<mM{kY++Eum~0HA$bC>%%pdy$W`=nxM^Ufo4+SH&M0ZEY7Od<#b)UERS>6j%L}wh8~!gA<TVa;Ubr;8CTE=!gD!UPaS2ZQV6O?WYRrHGnKuQM|vmxBC$O|1tdk4|{vjyo}&9m(eXl`B>a60Cu{n;3$9UM$=U>kD%;<6!8`$mv|`V`D#_HaDq6wc|OnkG&_d$>f|kr6t=A+V~rNYO%D%zxt?d#NKK+xbfD7;HhTdZf~)fK@aLnK#|-N5-BEP%?(oIy{>h71N70+tFaI-&|Gq!`SAYcv60R3skjXloRq5>NNtDj>1UCCJEuqIn);%zLM+X|VY@U`^*#!5tOy_X?O1>6wj8ZrkJy^|21bePe5>JtC8o^n>wOQe{z@vxbon|R5JFE+Behvo;)=kY2`hqKaZf<nC(gzcF8@|I?FQgkfeXt&vy(qhdEd+RUY;jF#iK)E|LeGsZlk7g<6nh4MTVB68IRf<bm@f>>B|nGNt1<xp`si-}a(tOrFc<_7XSaYh<l69+jULd3mR-DgJ93k*_rK+djxV$E)v|zDzyep;1#}0)U-Il=ZnFXo@AdvJpf9^}zrO?Z2}hND-pVnIYJ3SN;W*=!FP0f>a9X8(SVcfsDQ?L)y@o}H`vc2AzDySj*qmYoD5eX;-iWT?{!a^-XgN;llCGCL213Jt(RX%eEJ?yEnj}#^Ulywhj>e*>Qrx0)XGfK-F5rTcnfe>h(53oal<L<CPE|2yxVXEp{tD<@Y(O8XyCobCDL&5eaK29GS-DKdneK0~o-gkrJl{)I1oInTI)4l%FzZ!*zQ*;3y7;GCVX+_>3%_(QT`%ZX!hwPc&t+ulD@-+=&9a#q<ulXIGuWfI>N7Nb_Ux_QcwObQa)2AD+rsaca4@n}&xUhMw-U;Z*XMvo1;(F~xoU^MWsAnJjRbRmI4GIUV7U^zk5^Fq980k?s}0@fasZeO&_;r<(^xfumkY|CV$lx%Z<0;n!r<jdCX;0hZynyan_@MaBxU{~i$-{V<Dwh@U68Me#o!{V;?CdRynFd9Iezg!k2<}mGwgJ`)GnXeO@ACkLxzX{n`GDc9E{W+(w}i>yq=^TY{<p<=O8bW^g7J}WzDj<8^K(%s54%!J6#QJmBHa%M9)+B){FM&|Asy3WQQ>#Tz~|Q=2dMD?>jHwJUx7wygPjTYv+`KHdM5$SR;h3OJoM7AFhR`aRAq?O5jS5v$9MsR>gW5pQmF$aFbDIF&@ukCNhhY)fy&&AbQrYd)8X^+*O;>vK@Rhl|KPQ-p%nWOBd@UZ6_cUSi;FcbY2uQ*;`m@DW=DOhg&8%;v6u>hisxwfu7AIgY#S5c@t6{LgP8$ml%g}fD71-`ae;R2lga+fk)AvG=ukJJ<gJO23KQT#`5n$gs;kb=&u(|0kKx6YEuuKC9oIAbn3S~*mx>saMC*!UuAbCoP-jt3#^5F;r+T!XX~ugXBem|s1?7LI`6MeBS5{;6{2Z`qQ^c`gPtH39C2$J;`jU3sR8Vg@Uq0*Qx6`95q)|`#c!5_<KJF6W7!>9le5?RBWwnB5nm7B^_$1tE|0DkUE=^h<jWWb*%L_JPwvJ5EmO)c{ts;mcOKXi88BZ;+JoAbaFwsQBMiB1NAA7w3{o#V-{u)0^(wC5DW1_9GGdUQq4kA-^^#IET+O}OYVJ|Nd0H|K#(KNE{@G#TNPYj;hNDG59z^jFyQF*lqqHPcy{9k(6MT;zB5m^Mam;_JBXh!&#~IiKdMaFdP+kJl_P#$njs9g6J$hKbxl~OCzl5Q4!TZ5r(2EWp?VZx6Khm3oXf5~qqrHPiLtR53qA~2<d(}jt-VfQTC=;X&WA&-q_6?6(J|Fdmd^i?B82Y^r4<0={wHMav9wyVnLG`Gm1x%I}cr>B~@Ff~gFJhq7N?J^hwBiLTi}h+unCrb>z>oSD{;%`;@YPZ0AnGV~qtkQ0KRY}*Oy0eDa{`sHvw^CZtcs%YD!+R1>V;F8D40OS<F_wgoE$@~j~${?aRmGW=nA;R_;*oE@UIG9A^ME}ehPK?b%C86y-MD`J9_@&|HF#@a=bix{nwLUl9M+tkAOyaic7ceR`~`vk^{}XA>%+4KXK5s9n6|7(;P62V!MaC@?+nFHO-Nkiyd|qj!jw2uA%is28B7G59H7m)X|KH&n`al9w=bl#>IR|<TLUZ#2S{&UY?J4h084Q9Nk3)U?Z$WyE$;zKozS^zD}2+J;rK{Ta!Q|cY9Qcy+u@JZIku0;TZE|HXdTGw$%}qs$X)F;h+2q&N7aG7Y%2op<*%oltKlXD&x{Yaq|H`{_Qy&aQOZ1#anC@ShMz%zTHl&tF3loU2VM+Q1!pXPT<UHJHdsuomkg<*lKa*jewWM;PH<O!0-d)D}fRx`T`5^^3xJY#X-580X5O-b-MVb+HF4Zo5`$z1sEQbn4c^JS|UWNXY>97xDx4koL=X^nUx20uwKGTmBsJ%rPPPw)Npi2I<$jHCzDvHKI=UzS7RP2{0n1MUvPM<@q0dnr})JK=RCaLqZ#XQJbn4%ZNi&0qJ<sL@?|p178li}*(Z1=e&1Z{_~bud9&J<Y-HT^OTmkoLl9_L>-kiL6^E!F{_IR}a$P|+xIypXk^%gpKcX)C%8b03J+q2r?L4JO8c=Fr3BWgX|A6V`G$Kl_SXD?ojhNegk=BpRSuMSV1{*pXB{+)-!!-dON&QX|^jLU1&>YM1Lk>@J0Q&d4i6hx0$l!x~gj@dk2RQXuR{o=E;?-CXwFRQaN3t|Z?Fiuyc=my}?I={m=44%S_ae#uw?aRUPu4`VZcSkVk*9kAt$(vt~ULVUuV5BtAWV*tQkA8~w?3r9YHj?u`id2}!g`uL>U_499l7w*;iHdhO>k(>W5<LQsZA4N4zY4saBmK7CWckHqh2tDpBRzie+q<Vn$0Iq{{GqFP3YZ^BfQUj2^KC>oT9P5YhCo%qjy2v4If{5}9z8Up`8|31=GhSy^AQJt3DB-~jkk27UjO#$c=UeQi-vo>Xnzl>as0h^YO0<j&yL>e%Ys8oCfU-yEGGyi`Q_;F*|A-V<hgdOUtT<WcJ$gE7F5e8lWgG*@A%~K<QRwei@JhoRVE~~mX)Q&A)(sHUbPq&tQV2wr(B=IfmzS85ecZI$<?ybEC%jNY>fmz_JFDfbO2l+@(3f6cmc~tn@&2fJ31rQ00>NiBg8cxS22Ql-y!*52Y3cNo8zk1DuXvF?hdLVmi^c0eu;>ZE1@uSo>g$3ZF$ggu*&{i1I3C`Ah_cr7QxxLbgYDnd_|s;&g{6DR=@wpQ?0@~0k*M(9_QIAUMvQ$XtkstJhur6q>{v(w=$be%`UOJ1~!iPU>L@uq4jPum*sr6j;K14>f85LOLHQEfM@9)EQtE_*lLFdPL=;)$|tL0S*$CO2*5Yz>dw<!ME|PY{@z2YDR6)Zt}_bqGY5cG3E9nXrSDB8<RBYrSr~u0%8ONUldk6LrJzJLu%R_(A8o2SRuw|S$u%70^YGMWDG)^Sr9BesH5y;87gzoWtZ_06l(>6gZPL2TkY8Y_Cwwx9)jG)KdI?9bJJ1KM>zuTKIYOFi$3NUk>LfuG_0z53>><>0{&uT6I|)^tzul_tenMThz^xqER*`cDzI3hk?`Ob7SA9`=Grx%{cz6U!?<hq|`%RkWEFE7)1SvwLr^R`G6P@Q1KpUcc0mc%UUqCjfPdq!bxBTp^&mhim08xxmY90l&stOAvlt7oOSq}YEWme&7T`o`!E7VD%3ap4?>ZCvHMzRXUl;{x)prd4jH<74{9B@B8<uG|Tc=>oeBlH|SV0qo_u0Jj=Q59Jk9r8~$`7#F%1u7_k7Q9BVc>H==wgc&1Dd-wzDT^M{K>OuoJ_TNV$Kl|w6H%w$6zka}qAqUo5(pERQ<b6G9QJsX_IVY@`3gtALTd!r7wGp6w#WiW5(JtVnhx0eF&e@$3GF2^j8%S-qYh0CSe+7jcn01bu|VNfN^j#~kItX{rE8zOV-Kv-re9~axj_8t<ZU-XMLChT(ecr%!`CM-o<>aM4J?4aiB5MI#W$BZoY5H^n+58oMVtNme=s1+3Lu5qTpQ{YChn})BFiQK0Q!uOJiUYSFo99^aX(Qm1FZly>fvE)0VN^tf$~DfIKYA@z2HegQ4y;4zAd9^zA7t%3ApTv5vypefftT|bw7I6i<XzgBAcV}%~fWD#M*hu$2I<2dj0ZMA4S*s4A`J9O^C)Lb}Gw!nW0%tr1v^q+})&i_ClVW`RM=b3@*kR_1&}yvd=N{7tt>t_&yR^xP(*lp}^5+OA8{*bw8Q{Wf|!s@&j3K!0pu6kMGTMRB#e3ITxmxJ?l%hCtw0hEjAzA1t72JS7tzj?-26{8C`#TiN;R$T?3r8UXi5%TTghF42VJ|rM<7&q%U}+oE6mZs#srKS|G=(qAdG_AqkF+Y3NGhm{6u10m}=A3@{E}SXhEX%kaUQGomuM{}X3+aHjL=op<VxgIQ*b$lyxB@rCo%^{#D=JK{IPFZNzde0p^i5+;N|{Upi=-OM5wq#vuq>q2M;sK-hc`P-|-7a30SANV{j1~1WEZ53;>v1d_weh0I8z2U>Xe|v1V>A`!JmNt2BM85XvfbWHMM1<p$7e>ra3_ocB(wt@YiG%JR5WZ4)>Vc+)E(bySr8W_Zu6bb1C;AR3sOB|U7G;vn&l%Fu{5+dXU}&+>uWb!4Q#cZVrVYLE#E7)x78av$#E<1m{1`iuqDew8&nU#t5dn;+Z{#tBtmK;~Q^4fV0nkB{k+UEVy}C@wkK|wXBiW+~-*%DB)}FZYoJ1vr%{@M!h6mFNe(d@j&HukJ{=!(dT@h0a%M)8?tB00G!a<DqpD_Ku38uX-n}PBOa8`0BvC?b!Gj6;<(!sWE7z25BsxGA-d&Z}TPw34MQ$Mm?Gw{$Pm4v`Slh$4ovOIlencz93FxpS8c73ujUKC8fs{*E(5&0dU%^m5?kl-p^`XA|grpwVlh>Tg4mW?QG;P6T`pez7VM=>uX=TP>_&OAld2isUcEu@5I`O+i%JuK_;qAq#sezOQ%=fjs>X6FKb(WMPs)cWPz`a-HHe%1r58gIIL&N@1mFR(WaeQbZPUJA>A?|xLS&w-AvNQFvrGxZ8bH<v)6r${vxtBIKEs?R&FIyZ{5!NmX#)!`H?tt;6F!m%fQS{vs^aFb?H2SC}E*AbFmkWr&Do9jd=cFK0A9lH@;5n%S#upg+2hH)&aPQ$UD=Vb{u0^jd!!prnli3MZJ^Z}3j(AiNRBO!WdH2Ow%OVjC|y|)b~Z2A3A487ULyV=Cs79247guJ*hzm_E=WD^qcv$?fxa3ah&(s{6`+Tj6~s#uu40~Gt_Z?6&jyvQdu9o`d5XO|H~T0XsrP~gu@A@NWPMu@KQwvxrVw%UU_O%A|d`%oSh%b#6Gk!RI&b#U~BhO%0dfg~J(W9St*mvk1Ji`_y>_OQ`kk6!*3TgSoyF5E&0S}*#cp%Kouj$nQWBNnG+zWAfw(C*TLfLnrRypLkS9yyG##8Bk&-Z{^1mos4Gs_5yP*S{aVJ3d4)&*97dyEiY79G4h8^T3R!^Yi>-4G(1*{q2|kw18<UACDI7a#q1<eRH$u&+~Dg{w8a8YF?~Mczdgx3{X#pWbl8iOIuih5}!#YijUqL_t<?SBg2rWV|+uzNM9}sR4y+`vP4g1_lX4!*;9Bj@TThN%H=T#3o&0+@L6SV7P^me^Bf@_+!&af$Y{jFh-^~i&Hw9(Tsj^z`IcBVEVj;$RE~YNv0=5L$Xr$h;9CYtP6A8h&y_VedA0-*#k<45X*p-Mx&WkujfF}<*$<k+k<bLT^To3xls_H5#7jyeag1-iF&7sfQbzT#^1DivN}>@ZSRxFR;II=c?%GlqnoK@9i1tQgEri<_(}5wP2Bth5nT@3{bYgmehe$WsAI$;bBJqV!0w9-N>*bRDg>Cwu>Mo$vqL?`n-t02H&WrWR$O|Xh&tnY;V<x_2lK_UIIkBwX{YXn-JwS$q{Jw>T_|T41l(fDZl9ij57N&Fgk`UT^prN|h&u__?u(uIl&r&;d8!0WgYgO7dSw_beC~tYQ@(WsDpyu&2*9#!SnpyG)q*1@p%6;FwU?iDU(v*!PjWu(dq&Zql`o&ZjE0)SKX)EK8NLdaP!x|-WHz?H=mnKK_V?jiGNoYEXwcSLzzCa#J=^M<uzf8+YgnGqwj*SRv7yXx9N`jFwiMv@EO~@aArGRPH(A8}BG6OWfEXvIG^t%LBZ9UExsqw@kpt5j;%;{ZRvW$`j0QljhqFa5b@;qH#Mff=NVVWKS$XLAe1vKVyRg7T?(V)b4|2*Ilb}ge}0{&8JkP3GS*+;5h+HVOelE^n^HHYmW&@WLHasgKcP0pEid-3|k$&162qhp}rh)<rbvaAop67AsOI}2cp%c5xjRF?oHh<chb>STAQkb*IFo&c?%j*)!@#$t>>=s6Sb*UTn3)VyXWU?iKE5@#d^pzv&>;85i1$d*6PZx0mxiEm)G>ig33pR?5hp8Z)a%Sg5b@*E?HJI~}DX9wU_^!()^(9Y~%ORj(!X;N(B<94}vwAbyKsn=xaQX~`a<HX@Czy##QUieQuvFS(*u+0;-p3YK_JVw*<OC{i50X+|Ayfl7=JUU$X^9-N#%z!CL9alK-9W96))2l37!hOBPW8qv6+X3+WuFCp39x}Adl@}1@3jJ!0k6;wVZ1;d$c!NC>K%dkC(t^6~i}}wU4)ZQTZ4o@sXoZoJAciCss9=E+!4?1O+ux8VS;Hx}cjMW!w^5dpc!Wt5_z7$h`h1emW0I^Fh_n)8|3R$&z(b?<YHTOkwVHk2i(!LB--U;t|3Xl6z$d|H0(gBs+s>(umB|^GCR93D-aSEHe#Ta5WR7Ehw+Z?csRau(_LeJT4hy^#Yz&TA7||V&weHoL!D&YnYO4tHsho7iZHm-Nr*stZg<QR!`z%C}a)XcYdl#ivS-MD;(SQAy^p+r53k44dLiF#dm`@<p9Qb6!1rnUA<a~{TiaQpdj^ic~Kg8#`yIphcunqtxWdpH!Sgme4@a*1yadmvCB}64^d70v#FoWypxs_6I$Y}5hDIa~i{Pl`1lTvSn1pyesAj;GYQSlBCJp;X5dLRYVx1}X3P+CM7?_b~j#+*WItMo|CoJN1(Y4YF!1wBK4&3fH`jqE27&K2qNOT?JdR8K~r;sV(Wh>=ZK_mK<SLneTLR|_G}V7^bY*{lgtxHZnViTL+bj$LoPj_iH1wW&nRp}3FeI9p>Yl9!(P8s5=}PM&_dVvL`FnphTP;E`oZ?z(n1u=uF=q&Wm}us_G<CHp3N>ZzmIbDv$c%tI~8O4wqusrgmpb9QJ1LpqV-JEnt^Er<&gcX@LM9_UrkWk|oGynmef0Fl6n{`Sj@r@xrB6yqKspw>$uOb{j!aXtH>ud;ELlNm?qkCC5AyqWX#wx4pqg^3a?7K&cKxkS!XdJXS`d<>~Mg9`8s=NBOUxwT-LC-W*@AnQI~&#JsnudCV4cu1Re7u~|9@y;xfy2PO%RrvbNi3bf0G7%C2I#h5D$XMq1+1o>=*=Ff_0S{P7bnnFqo+zjXy(}$qLspvbf^9zmG!}w|)&BXr6IL_8F!%N!-VXQnqQ6J*^GY`>{mEvxW7Y)$q^G|<JLKNUZ!(G>KYH~2BQj|2<O}lhKzuySFV<|`<iP^Wx1N&(Jc{1VkRGDZpczG4!E%<s2*E8xam^a-hOm={LqJOYQS`83L$?(;2Af}fS@Y}9YyORj*}`khCVk{LZlOL2Hz7!a!o>i0X;Js#<@(G4y-JrTfn%22AQR$eFYlJ9_bbT?Mp;P2B65n-3k<|j1+cTWoRKz!hUz#N(Ncu|G-J!uN1g@`VkqQ^S}0|!BZ2_2mIPdOot1&5a^Ht`qIny_*&~iy=;qIW=23Nypd8}`XzOn01J<=HbMXb7lPeGFzllS#ZP10_%S+r(jF=M7duF6wMVSS5IJix(bD%NTiyRe+Njx9$5W2M%=zlyQO<!J;I`}nUxjJAQIQsy{+{{4qK6^5|+5k3G-wa#H^R|MlD{qFYC4K0N-3T0GAi<VY!A96gFX|5mkM2phAEU^Q8KgyFD-Tg9LV;C?Kq<%=_|Etp<wU@q&ph%eG(Evc!2{7&-(5H~nii=ET7R-~HHyiU&!Teh8C)U97SD<cl%zj|H*ZPi_&@COyX?xL)(gR6yGDHAiW)L{rZfjw3bq-v?y1GYZ*^P@^M0R0;I)!d@NlqaPZE_nn(hp(AIV_QoD|U|?rJ_$rmYfS%T{WXk!;XG;7qI+z#%g^3OqZ~k7{`pUN2UI?bM^8RkAgVTc0PV(H6H{D@Ak<RR7<RQO(Emthy{F2Hmp4D#74jG5haHJc&kcuzE;_T9^X3d212S0&*+`a4%B-wglRsa5QMIl3t)7bTn)qKG_B}PlD{oKW{>vPRDU`K;Th~@-BF3Q9TQs^G$}Xd&bvNsA;B#NT|*h99P*LG7F`0TtE){7SG#{BV}g}tjQo7px=V=@^SnZqL>HLjh3@@8GVQChynRdc|$mE6v~OhrfH}i()Hq1@#AQCb_QJi3}{)5LBzBYn_xxOv<LR0*r%VMcoACu!H2=j<w|?ZrK@ytk;OrPU6kkXI3rs;ZngnAo@G(`CmA|;k579qIyjAfqCXFM^w^(9jsN?Gwwj^yI|Pg-9||h8=p5jC0of-+>e0>Z36Y5uof`?IN9OF>9jxWhGK5uzCLIPEJmyOz6WP`$gXaJRu^2UE=sdT&wS-m~o<={KuALoT62WO2EV7$~MQ}cly;C#%s2}K~S1a2;b+Mv7D=Q9)E=J|`<g_*;*+^Xt_wPUr<+Q#Pv)NoW0`0u(+YtR%#0TQb_X!qT^+90+Yb2!p>g=Px%2a5UxmJjai{|NOO%N_Gr0QYY!yzrVJsPUI8L|t&I~G3lo>$@@2{RAtKuBTxaga)lhk^X$L<ua%VU^x&6`III8VY#$eW%c2y10x1q6tIPwOuw<W7p>|+?Zn&bsc9;)!uUiYyd!EhPvTLd_%0EmPc9c6UBl{kkF6K&-*MS+IO0kJmkFtAQZWS_&f7aj>$nEkIth#e@T5P!-M@(f7$#x`v>1|?CJXrYqx4yJX$Jou^dAuIYz<|&Q}hegHD<PtI>n03-OP3#M9{d>Ba`D&50iGp&67FRndFAc@V~^xhVao+Q8TbfD9u>K)d4G(=>~2cQBr%^JR?Q)X@-Fi2-o-H{QVQEnvIa1pQcw=@es}1Qy>7rzWJ(I|^<Z9sPdvI!e#WVz#cbKETPva%@V9(_*Dr3<}O1sT$U+ZHzh4@KiA*7z2d$cjYBU=!~tPAWC<SJfH$-Xc!>yOx3|aoN<8I19If-Ok8#_Ft+C*%7zQ)nX<;R{Z;ETWVt_%RK#FfMRt*{&M`)@wno9>2*qcp$^Vffcc!8#6KALA!f}pEOU};hI{V1V+QQR$l`V}@i*`bJ6>3PVkw<!&PS7%4p^fm5u%YCF)TKCEWDZBRcAgyocQx8`Y;3B_?!cpkQ53|g71_SvRoZhL`~kMY*Rm%AG*#E~fD&FtbqwvkRNKZIyXW7XeRu)bc@A$OM&-r7_Ier=P|0X>1=j?w3S1%yPiWl}93d@O7&)2y^LhN14nMdVO<1Vql3QV8AY?GuD4PH8ccU%-hXy0S&6VqO@haz&;o+@VLK>R+b*-LS_O6D%?7DaoyYHEsc;NZN#e6LJD{sIzQU8no7k#s13!-c*D7NA0IabK5l2MkZ1Apzl_ONtB8rw!fP8!~~1H?9Vv=OqraAKx#^zeHGN-;)e#v)-bE|zymOyJ*l$VTB5!=d=~x?vF9lM6&F6>4BaJfmnSsEx)KcX~NkIz|RuEC_dl=i0m~TJHyj0Vq4}4$3v$WVz7jU~64!sjjU)c)jltjZ<F2SFFqss`B9}Ru=0Nomc@sQnW2L9g1SdOpMmbMXEvp`cXSM7ONGMLwgfj+XI)12b<*>dOz+!^Q^bb#{Hk**AfGbF?kl)5Px+i1Jt<2Lc9&oA||d;a=cl5ky@4!P0oQTzqI3<JZ@jxx5^ez?C#j`?fuWa=;ZVu8nOu~y^}nRlC9a;<<6oAxMO(qZ}p4(ZK(_m^)AIEMV-KFy+R-P6us@IWTlCrf!qyBm)^?jb0kLj=CLC)gmgLJ?r8boslk^Sf6|p4aA000-x44z;$hZ*T&Dt|*B@mS;Zd8T2yRv^E*vG2>;{U^9ofpIzZ;hSbnTOR@4OE|RgZD=O2SeDm_GiqN#IBuOp<>N|IVgBeaAgSG>4MjgZ4}NXFB3OQCE|Sab2Quffe4f#C$90b6K+XlBnS5=cDIu-W>@&hQ0+Jm19jaEV!^Mj9OgA5E&|8D<=09ucb<TxyGd>9Ow^g{?GYB^=J9F#}+$gZM6p3v&U-~w647OwEe+n+-NRh-9KR3Ki66IA&WOL$fV<B5M(|UpO+t0MCT0dI;t7&Hjz=Nji|i@UZ72O;piMVMsm<uUCio}SY9wn*toHQGhNSSF~RBDBX*(UrEG@!sm_7Xf+z_b-U0tc(pquEag30yqeL}a$@*HB+7w*k-nZVyHyH55L%zW1<Rv|IKc8aI4d9q=<kLL8V|(o5lf!o>@RL)hSQHDJ^pRBzTTA>j0=#TSiNbhh#nX%8396rX|D=Puo%hWp`ihc0BFdVX(YY?j)?$e4BD@0=v^W>mi}DDuv#N4szD4|LUvhkiHC-j1iL8uN4=?cG=*x^70IYP@f9P3TnniZ@1;%9vwiO9nF)B9p*9DA@|F%T@l-4vljB@zMyh5mmNpK%Vd_vyKX+6dDHLGi^2+)jM4TN6LSlY>T0Yre^@qeSgm>w*e&uu^<N74)bvl@rn8P*Jh2Y1_adp?i{QC*mcpz5z5dq<0Ep*2cgK!&~Ji=e|oaI=~Y8)!36HaqA>fkUSbR4lc<loFP$^@cvRBakVdaA+X@{A<^ECyJ41GCbukr?&F$>`aXcuAyD=6j(sQ6=-8!svFAcJaO#N4m#{sV3trE!x=WcdjNcuEoZ4RBwn%^l^P|MtuhYhg9KMk`m-Vg1pRx&gQY6^I7*B@_2w98gTe~l1q9(;Ay#(O#yT13`BOT^k>aHS&B#OnU3t(z!KME!y$oCRI7%&Z#LQxbG!afyRKlo}XKOS<AoLThffyNu*=BPR9Sw;uN=PN-r-Eg|1R0f<gXh$7QFIXT%NwOY>C9E>XmtwuyrI#!74M2fU~GzD%))$IC;HN)i=jG`+s*RVmWA;zEy_d)=1&~7R|FL<`Q~Y+>U-xcVAzn#+7lars)o!HQtv&_Ta3s{X)W|+PlK~)N1hKKp1Li8S*BXNy7sdq2xnww@W72ArL<vTO4GEj$FPYt>54-f;eWdA4T*bI28?0t=w<!&5W=|W{ZMps(>qdoCv*c9hvKWeD^_GtAV+331=BrV5gA~=_L}&whVy9D*Wv7J*D6-C1-y?0;(5lnA`XQ95Hos=+>n*@B2-9v^dgD{GRrQ~@txw?y6C{gTPU$Wn7J}ftTuo>Ulr+OjL!zDBG@2>&ahZarKLFhNnn!oLJ!&{nMhec)+1`IVN&IKxtyV~J>P*{x@)^^w~YZ$HY}Awo?#Oc;4tv&OK%#V1QLg-Dpr;+k7G5xh8b5!NZZ7*ntmLXl{hMMMEVY)*6=#k$MtodI3E{X`fNC%pGMT<V8k#k+}MM{RAC%dt4{dovK!nqbrM_N2&{sff!_r8OUo8W2T0pA?Adg*8W2PIF1n=TMT0%GRKcPyl(ioT)rF5(FaH6O`=`YMm0~HIc=ahTf-M?4Afa6!1)vi`F^T{sk4+4Lt3es(Q<ey&B2&^*Bjakmz(k$nRYp3>nbG<UzP=WqO<tm+`_x1xUnC(ZZqPX=4T?zG1S0yZ->>bbR=JRl>0w@793R>EH}Yv}B%xaEukgm&w!p<gy5scDmS6vy6~3ET4fAoRf_L)^jDG*`U6~HTKCOveSmc6y2*oqenpYBvV&oNcaO?}7>AErMoafmrpYtrMB4V5HeSdap5-3^k^Z_6pgO}`3S@Kx(6OA^NokNxdV8CyR3R^`l_f>2d@k_D>Mhiv4_;ApeV#=(Z06-SBf<Dp5#R_z%1Q7JJr)t?&lua8W&5ZVtPFQ3Yv-~0loJa!iJ{wF(3|~9<gwPWbL<VJC5egUVXfjR!nR;lLws+j@MaxTWRr20)x~aSxPBoEAIZ7I*kqFU7HiVxdC2vs(068$)l&-6R_`W9eF)c<C;Q71yC;g024SYsS@1d8DC)51QnYUF|z-`#R3eL?}tKmU3D`KOzEmp-^8MkF^b==SO6>>-9GrAt?YInbBoUIi6#Ru0SdaS{^w=mtHM-F9VGqX-t=D51gO>2+sf_Bq9&|iIeby>P){q-4L8wiflVE?{wg5elc{@4OB8y&J8?=fYCY$8O>=L<vsPq4cibcI5vQHb?yMIjx@y30@zXzxX#>&1eKjIcb0mNC5bX1-fvs`ZLuwu|cHVwEm1LjamhVaTvuC)MD$a1YrT5~I3e@OX>s3m6x7(anIwTv#7j1egdCnlN;mLpnP{F{nwTbqBQ40$se8Jgc*_6Uy-fvyA`#XiuKN;ohE%8fV+ZC4ps1my%J{JPgayeHu9eyf|l8CZs4n!zBqvSu0BFR*s_D_}KNLYTk>IfTXYf;MK%P?sKDgu?8Br&W_>sQ~I-0$8R#{bZsd8tLEJ>hNHJ67Xf@D{&g7X%_vv{jd$7<=LRgAX6L8m$BBBO+IKB84Z@>5vw=p&5S>=75o~{WT9<vgC3L~^p0`1Uc=xB$^VXtQ#|{Fl-cLD5ml%vA=1$#hrt@!<byYgPqI#t=en;kvEJfz0ZbQ0%BLiexF96+mvmb!n&vW3PWeQl*i_rAYGXJlG?Mm|L(S8u!Q{_e*eyRbpaapPVb~xV`Vb<8KdvHxJEDpWlNpRmnR5l~V6sun6ArcIFc7`kxTbP`krF7EBP!#2M=sQK?fszonrE-+F$JG@v3lLBam<=)T1Ke*7f4G+{WrUF;#@K`@2tzbuhIWk5ZQ#3oq<K$-Sp(KWr!`LeKB)4mqsg$jtA!QKUV8YjYQu--QllTNIN@%v=~5;=_pxAnE3xnJjkkkLH-wf3P-GKPGJ5SrKiFykD|ECZ??J<mCs81DG=}a)qY*2}Y%LNl3Mtk~q{rq)yA%!#5cO4hk#Q;=O`V{M08bi+I0333b;>mr5mX@6VJ!GycZ?DCCf$h0C_Bc5c1{V?JI*Vp<3YzYM?Krv$}+VI9vc<;IS!rzk@UzZxF&;N+>6iH?`+aXlM?b7husy(;%|;wpaS7K29_AdcFtyPhW1G3J>ftDw##}gPINcCt~jo;W{G2Ft5|-@)@T)n#nCla^0*!?n{07P$LJpeW^Hll+Sl5TuEw#5I8cif$4odAxJ<vI>CHGjx9?Ov#S`nWRM4lPoM<#(m<kcUR$(fih~5izH|8p+rdRh0&89-X#aFh7#hM4k?m``(ipHjI=dVttC1fvDST+@it%ajd-pyJg?pQ^JZ=;UFrq#a~*8}S#c8|XEfLg{Gbd(5h2Uv+K!%75$Sb5xq0K2kB1#^|V%_Qn~Cu++>O=%p85LV#2K7(?FCQ|SW2FFzAPriwcREToAGGac5Ml`G~#RIoq;&0`m)HqHRj)MY*5r4b<em{z-<_Hs9s(_<ZJ}#E1F)e}0=Ad!te2)xe0Op%FufCnYF?sm#uRmj8GtvsPdKIH$d9n960rs@gk>8#VWKeq96y{X|7*(!Z2md`3Lry|T)cf*J+iJ0h%>Zg#jpI<r{EEHBisP`<b?SsRL`$m&>@&E_s_i@&+F;iw9B=~ne@Uzud6Pg?yvn=#CI_6P?8C5Y7M?po6o-W>VGZh>(S!>l>IK5)`SGqWV7*0$DjY{aBamLHh1S!-`0%&f0Y=gDVuE<qO}0VTVhrdiU6wk&)&jb`C9OaebtGP1hlj@TCM4zxJ&_ueekyY@)Dt|}5bb3P<amWt2FhK`R(kl+zx4++Wx3t%<D>60MvR1I<vR_}w*}08AiGbZ^Xc%hZkA3f2^2_NHHCFfZ~Nu?yh4-IoI`?Q*q%Y{(Kq+j%tY*N?NoPSf9v0AYxW*BFs0y^+)LX{*Mb9Dk6#S7<0ye60`wph=1>4=uiK&Qiv3tFv<-lYj%Vaz^i0s5(2NH1gbPCs|7kV;)1qZ`a74=_KZHsam+ZQ1y&zW>$SVPfV+b{;*&qx6Fo2sm2ksfg836Ko5^O&-aIY=6*IyBCIWAV2Ar3hpVf-fm8Ib4_xq}$bLw~=^_S!(UmVd6Q7=T-QD@*aK#f4Xu50*RA2T}-kxFrEdldIDp<tg_+2JkisVs8qNBQQ7%fs2&&3j{cg&OO^1IXjlf_p@8(5H;xFfpq0|^|zEaHMeUOJJrpq5of)JSE}b!x5P?XKsU0NvMy6F=0{o>T6e0~woQ0w_K=D}`{pLgMc0PiwvjT%!D0I+wMTV9O%1p*aSWMa1tbz}CdHSYU>0{hAQEzNngqn9(##T_cnYlG5Vsq4gMfxz_f^0X!>ZffQuch!sUDf17HdD!zna`~qTsYg`j<_^xug3MSH4DP1n+-a-r6Q9Q`04Fn)XI#!Hu`nxZjea--C$5&0gf!n0ka{JOJ2^fZEl(qZPYsJkFLCrxxLx$ic*BcP8IlJ@6?8uE*>M)Qjh-K<?ul6E$BLA(Ap~KVRoFVY2*?sV#(_xr0dvD}908qz%hgscm5t+1A|lo7~_v`I$bG0Rz;}OPX$yfOtu<{p8`2w5ukNp>3aKRV~Kt*kP0WWHjLf*%<AE<fnFi5z5<<Rw+I=e7wE!mD!xa5X(TjPLe!wmgeMPsWy+4bm*BC+tDJsJv1)F<>i;Tn{IM8Rbd*~85n0@n(4oNcjUSRi!ZQFQZiqcRpjKDK*z-iS<<Sg%*0?SK_~DoF2J&$a>dSw%1K{PCvKaGiU@j%IWKBCR|Xzo#4{T;`O{uH?zEcHJhg9?!PU*06>Gcn&j+<cC9wfz#~$5ea7*dZ$u4eJ8^!IiMO<Gj;w)j<H*B5@XcMj;yqU%Z*Uk!OXQ}B&ZmzdMO&9ywlQ(ffW9#L;NAO+ehA?WhrNuS6Xv25{;u=-OmdseZRoZkzTjR1i&A4Y<vsAyS+t0w(O_*-Fn=nV_CX5bS%{R2Q+1*PE>3>l5INM!)Hz;(Ler<GeyP(?ULr7gSx#Q7T4a@|jsTK0f*uBiE)UXMDustiHbxb`t3ZgQftjU3{skMq)U2AK|4~J@it)mWgJrn1xrg^CcVAnCM?Ny_mefk;Kw`T@n{fzqjHMGR$*M)6p(iUFNG!Y|T)u<`ZTpI<`qker6l8lA^`A)hCtY4B*a4TEX>#X4W8fk1o?@^c+JI+e4r83lQ{Y)_qTcDlR(%YjXEwBj^p-Luhc2FC#(gvU%D{iRaqA`Dt#HfP0>J)GAOF(=iyU{sd7`_JrU0Pj*&zM`~1EDqe_`&-?haSr5U+<e5AbeIoy&5=A_=^AdU-b6lcl}3WEj%PnO)T(g8{7{y%^zMZ!8lMwp)L2jMRRlxYzoAyXYe*uGk?@PIKHZ1&#%AtS^I8<)FYxfR^~Y-#$Qh|*<vVlmbiBZc0{gs$dHxFijnlK!5rx-DAp4WT7mUz(Y>t1KcyWz#&QS3bh-t*DyY~MNMHjs;=$7X8WYjQa-ym?@!z^8>5T1y4qyJ)ez!aLM@AiM3Si1h#QEXbdPOqH-(naDMBlJms7vD?*FE8Z@GFNFk%YC~nrqA1S#Ucjl~0V%c{0scWyLX((7_Uyi4q+A`a5^$IJn42$h$=bW4X1EC;+JD1UHEP#M~UGQ4Ehii5Ldj?ZcR*5$AU!?9;cX{#a5@XUWjBp3Tr%rmP-dNQ~&$-QOvFDCVz2YcL8f#pASxsg`H-Je`&3IjY49rmsk|hs;<~V9LmeYd$l60kKZHb#4c0xBPd>CC)^U<X9~f`J1D#&>uY~6*Jn-$X&e3?j(Dj8;Bh<q2U^s+d=lov?3B>Jt)9{`Y1}JlQh6rIl&U66j{%8rXc(<&;_mtWWh%eQx%T*JbJPZwaXNzp7I&iAzkW$;E@}~w{CcoKd-Jy2{n@pFFhVn&TOvG4ql;~ngDeXJ<@*wEcTqotjs<+h`O9VEoF#IP7m0e*{-G881|xlS9x>wg)nfS*Ix(&1LE^-Y-Y%=T#8xa5fvSH-{m&0L)8J>T-U2p?-~c~G*~(ignoL1$J&bUe?WJG1J&nIOt#{Cg8{7qa^Lgx&d-jeZ>oP|NYjkq=r15aG1q%VR4Ck8%o6r(Eh$o&sLMOcFBa<c%2Ue2Ru#+sESti4yIB?RGNM~~^d~7M{snd(e+=AZyArL%m5v_ny2B7UjHE3PZZuD)%AdI(GDBTE_EF(`CxSjcH&VfKySn<r2Xy5G-Ut99jj1EB*I|xu%zt3m*I4H807lBf5$XPlojTYHNwQV&Ub)cV=1L~S=L6!S;BXgY1~yrbt3>0rSS~VOc9MFIXeiwlk0}&hA)3Xv>!5~C@xb`GEfAsJMb~+Wt{7eQ*7(Wi%tJxOQyPVPxi3g=ZapPPZjkhE@MUYz-3c^RKYg^|syPkPHb8C_{)Sy7ObSr6mXu&xPV@T?!rqT>Jvab^KQqeSkDm`Nve`Pe3y1L~5L7ii&@Io-wx3^XsRUC$tay<kmkQd3^g*LysvK}jnLzh6NXkkJ2p4T#vRL{W&?n!b3?J<=;>MVDF{~_6_*VzyAsc5ZdJo;WI6$T#Z<MVTY$N-qUwg(Q^V7@Zqo%CP)?zwA^~INmMpX|5TuxGDeEFHNl_dgu^sjBPrgyA>K^kYV*Hru-nw&O!HA-YoC%`d4>DEIVL#Y#|xvPR8BMKY#hP_`-8v>cx2QjP<uH-~^8yzFu&@Q(7L*4BGQ3!LAN%i@j3~op#DEzVSc&kU%?(ROSG3st|ZfceW$A-puN#Ph)#%mxt!2djmY%~54B5#T}h5>#TlYTLkcsXo_7IP@EZtJMrAWfG0TV16!bnS+20}++nYtkO$%S*eWNHltGXoYoBl<@j0Oj{4-)V{E{JqHDfdOw6@-B?dOY{_RjsJyr0JKc7}NW{59^jNmvp*_^_Uj=WHlMX9yx>g7BgAUEv835B^Eo~#^73-QS>jtXJw}%XD+4#CQX1412Y*|$(+QTI`ET)&ZWSa0Vr}w<&SQ;>n{=*nY16=RnBHOAxI30huH{>fLjC&!CuYfqh$m6Treah-<AsY8S{_%wf;cFlTJrMZM{ZWHHNs3Om{lmp~hW(>zzGtVL-LHsh9P);QHD8K+lq$`7!9%u7o8Rwnh_=>u+!B^4GYFYqk1&iFEmW8}7?ZQ0ixp#?a$bnd0IOjwX_hxJ^mS|uaOOpQ+a{U}KCXe-ZpLI=7D~ffFREfqh>i$i#IJWk1B*#Dhm8~S?(!;MF4?CcFT>eP?in?W$#8d2j6oY62se0sZsF5-P>bSKk8{@KG`Su5v)LRcFud)^n9D|V6$0s@t$SQ@FDBb2io%J>)^x2Mqr5s@ewn!DREc%BmHl8f@~r4o$<9V&L9W=nhxfBHUgh^3;YQ9Zvctg@NVw2ADHt#Z$xsd%9F3doTga-0BeYt#ELJu!*LiAV4NL0;U>{(S9r_%~{P6ovlsO<^CG8b42*gypl#vn##RJVHR`IuG2yYS=5hZ||#wduHa!ebHD1-e_-JGGV@h44`Nm@HACjlc(-qYGLTW%t17UO_1PCj9WN;mAurgi%#BsEB?a9V@J4vXvIg%+blhm^TY>L}qevLGatdt<lHgrH5Ys*R>cG$8Z4(ekS5s`RRuaJ6RU^tm1RR!$AwG+X<9HAASR>TW;mW@D&)5w?cf(&ot=;E>gmcMR?`;IQqJ4dWh<mTMSgdF<8Rt~Rr%+s*b+_V~89h{9$1Y?~<icKtWCib|?~id__w3IBY{sJ0xKKlusuv>6o!zN<E;pX8jAJH9PaqrsH24%*VHC}cpydWe67{m^Hc-XxW6F5{7zFxnbb*<gl&uIpnwOH?%$0#cG$b`NTbS1tH2R;pGj*W&UkQuscmMBng#eSv|0VM>rtahHfmmhH2f1Ar6}_P_J#Uled4;VyLb#3Uz!%<Bsr+;Ak~-cEik4ZbJ$b8<tVl0ah{;F89O=vYVp<5#f~_E2QlJ->pNt_Lgaf0=h&yW{q;pT3l=Y%mf!>!XytO~?w%&K&X@byjG`J-dstKiBznN`XGeF&ZV3eOXN)Qq|4_xm#VrI*_SWEd-P9<w?!3N|{hmb|I8d@ev|Qy1sE|XNDdJa#8GUirFM9D<v95^PQLzVa{?S*!ws6Vp7}~Icm=gh|;ZE<}E*KR!H*H3gcr8y0p&%b4~1NqPQ!rt}KEq10>X)!Z0Zi;qs@w!6R)Lh8|IvKM?jU#~edI>d#?CfE&TxLsXbj9?8t;yCo4JXN*MIhsmd84Eb@Yt88{RXo*zsSV-Ai5-=w-^rm@b#~~KylnI9Kx@8ZIUbJ3duAk`l!?!&<i_@6=f;ju`6xI*DR4Bwh9qCo36YG+_RMFSY27P#i7uH*F{Sf%H%M=q$t)lEUC3#mI5FqK${|gusCcuSWIRQYoErxHaK-{>CVshDrBqv>!z*bJRqpZbt=9^8ZEahUW*6u-fp`e#CuvQ9~Th*c>o+`q@PVC{KUq!BlT!4b#)Y%Dtv`o*XZ_<S*yX(5p$d!}Vi5M$MsQboQ>E+xQl3w1qUJd)UO3dp*GPfgVy26rg6~?mOXU$=jch_IgiKm1i_f9`WT4R*L^VyXUcTb&&gdX$rO>|_mT5L_o!W5Pjrxj!5*s(-J9VwbIGA1mu%(r`Dya+A6m0(=r{6?SwvtcMz^*G6?Az1zMJC<>K@M7QTjs?M`EBN+e#wenVGzspZue)N43b9cE?*s7^9h>xH_IN52;|Rxw)Yl);LG%YnZFgGoZ}w_;Fy!^k>}F68f%-WQNIfK|mVxm34$_$(0Tc?X+N?U$@sF$oMJB6hSk&4J%8htF^$WI@gLZh0M~z~F`kbAJ0G#BJ=72kjQZQ>7dls!2LQFQ<5Zn?A<FxLG2$k-=A>6JP!fZ~^(M^8(EP;}pF~I6`B6i7pZvu|{UZ~UHeezC<bZ)~o(s}u}*R<!omem=_B5jDGTYnE>uJ0Yj^)%FAqE)3<>uvz0xy$7-lU5a!IV^*+u2u-{!qq^8agYbYsT2SLdo)6v8V;a^aur$-9jUj~sey$f4lixvU(NrHuL9A4C2rVW7cJo0zB5NG7JB#Qh^-I<U(Ou^HO`@p{yX!}Rk6HM{O&_*dK8Z#(X5c2aOIH|G8op&<E)+U|NXi1hUmr|@`A>5>IS<AIZN3L{Hv3<MvPXXfEvR%S_jK8OTe~IfpI7FlU}6AV-ijF?&$F4i{qnbY<VMb+Nd)}MQ9F~qzvVN^ruQur~?ti<kHi}F~SyHbp<2`lr(!uEE!wZ^;NuiVIP<IlJmj<IY5!tN=e_+izq*AAggD(LN{|9$*NdiTv82UP?c$|#G13Ljpm$QMhuP|0`u%lg?@&1&kM!gR}Y2~=>d3fWU<izy0pZYVy{h<>D8jRu~P#`)HhxXvg${0OtaeI_RVmfI1xC9xYzbY>R%Z!#gZI~$Y>n(#q|1#2U|%=chnH)@a_Pqj65m|E-ZjJuJb#NV!{~;=)O_FRFZGIvD0D>c-N)y*ew1jFd3`IH!4jyQTOM-ZcIx1z(R*6Ac`HF_7;#ru|}bX;fpcD^CBxtoo?C+#iC9CY}nIcF~w*D6pURWyv<W;R#yg;CMAOcX!hv|(4;pg(Nge)0Y{B6DTK{&Ci@w5Eo2QbNh6FGVH6{f(H@G-(phN<#{}`CRZZBBsw}fr6;R9dqE5?0LWvb~6wLTxBvI=B|Fob3-x%a_xjHht*aS<vfe@NjoIaIcz+gHP{p<e;F$x~~L~zuI|570s=??~2(+XRBssHJl=&c>b8#fG}6SVzuULnNs0XZT!vJ^xZZzt*i=Il_kNb-{*M2aXfWx1m~68uDQyc^U;^V<CZD8qu5POj4hiV`jBw@Y!kB^#DvKF=$}J({J&I!hD>W4_pOCm_FYeJ{_~(<!|#hGC#cLQ@L6a}HR69lRb4Em#I08$2ByzJ8{vGGgO+%8P<9D`F@N<&#;NSllkZR1~n+C<geyLjLwm^seCCnY4I+aYg5~BgrQR&LzpH-IcUpB5_jDv1&d{IcDk^Um53R=V;kSD4!Ey^8`0l=!abjD%OV=ex8@~7Lg!iESnw0B1NZ}cL(<gu=m-*$T}!KUtdQMuoo5r)HmDdt|q(C<MKaZ86Q0Lx}cdkQ*eNEP=V9a#%yV&KJ|xaJVu)W#|O8nP4IdNi%CQU@*sN_ohP?BA<I1Ps|qo0mmvdL88M*_U>I~VbWo4ue86H_7SIw|m?5gA=vI|bUqDD)`$AMj&}hXKVY4WvgNXFBM7(+u(R()<A{&ERLplGYMA-7gtZ+Ed0Sg|>k#A#g4>|NdSz!BBgeffo-I$i78y)ChHQ_FipC7Ubr&Cv;Sa#^jZ0bhDvhMTfM<HeE>#o5wwQrL;*vI^I3nvGB@_jEUn`w;Vqc_J9oV`_!_`Q3eqtU_BPE6=s0H$I{;dEJ~IqL`+DdHEPp`$(0G!S|1=BL<Vl9cepNzLacj>M26^72OM>;{=LjrKzXgHoDLG7N290$M0()`J$4N{>)LazMoznHpY`!lWT5u^?(Cud)^M?trK$qD3_Vej+_Du0!~`Qu(e_7ZVtDvGX}H{T6vXJZ4iH-6|OL*74ofyD!h(B)ZG$q4mFx=>JQ_{#%Ru|H<P1ri)0?Zz3HB(R$LHXH2<8csywUpvL-_nSBj8dLl9J03IovLCcbu^RV;Tvi(Le{w5LrJ(WCL?0(M`^m{vlOWG>a_phAIgEd|`T0#xg?j(TiT!Q1lNR0LX!EJMjXgn%`1~Zqg)F%)&Kei^()y`S2YnF*HM!_dytDnC0%qauq)9zqd+{F7`Cw5s4nn-gT=Jh*eqA4NU`_!eR4^R?gCwD*_VEM5tKL+KRGYn&#iY!~=PxAM7GX(!ohe%!PJ&YL~M`>dx@b4U^yW7}PU17)PfnG(mvWlx8dC#A<zEvTE@MGNRqkzw=9x9B_>c|>iy`iRs_2ENM)S^3phYS7*tPjz`&Bs+eQAGi-H4W{jQB6_M9i;G>>rTRIO!qcf`*IIe=orr&PRc~h8P>v|ggYE#PH5{eXj|8D*Q{}PJex20PNTli8bo)eSq*7b`x;6?kC=MPk3RZU7&%8!7+u=5&x2u&ogl^ju*+`4aLngq5ZFfSeiJ>{QhQALl6ihYmbydo6?OE;I-=DG0QDsK9P9^fVYaG(gzfe?!W>GZtmJccK=MHt2Wz>Udx06x#DHf;d)l4jRN>r!ZHOb*^V3-mby$lUrKHlS_%1zLydiXkv{MK@f|fD$DbYem4&kVcIs+D+h|_@n#S-OS_q^q?I`gx13{c2RlHR)0LBsMauL0HLKc9!_uaX7DDYAwb&O-T*d}dKP=z9f(E+`E)fD2aH(hDyi(0F58rS-1Q%p|lh6ZNvM34~#Y=W>O}BoTQqG{WR`Mk2>Z=XpR^&7z8w*JC|+l9uyfroFym;Ve0^7_i};ST9&{#etYO#Q)R^Y03~ejH4yD#lT-gRbOWd41sM#wu*D3)2;coCXVoHNNx!?wi<DL6$1@-&~k2+`o`mG9o3Ed?Ow-Aq`Gn@4p=_lQ5%Ze*yEM=#Jimcr4iNKNu?FEn`%z#{naI(vQJGWP`W>FX$>Rqtg0@)*Me@#_R~eSrKnr6q|b;4HShShrP!apdHwW=^AW`BQfXTO@tP~;1<GT1_QR<kR8b}&zgyxBKTpw^9sXXDdKv8-$fh)ywNOI%Y35c%`lM``HN7nn4w}Ut+(U3-5FxqJ3{SVST^#;ngcx{Cy5Wag?eRa&a?v?iW(fCxbx5R;w8;_d$El}ZJADxm@f#U}8Q15D5t>(?JGJEqsx&YwqlHzI^kfPa4<ZZXq(IT6eUsgs6_|(!HqV)k2ovOWItPZc6Ly;OR-)5iEsnP0QdS%f+h<!CfR^=t$KhQAkdS=qh;f!^?VJ7U^ku@SY-x7)f3F8xi-*_$Q9F)S;|<S!yk2j#Rh+(HL(j$|U>4GA1K#bnY4`|R;A8P+Mi<-Z_r}$2LLJDg_IkR1vp@a7DulF83XlO>QxvIiNXU-F{GHTMZbD*;?37LOTZ<U6!$+B;;lzG2gE-Lroch9GwX7=fp%=6!2*WH2*3r{LhgCIzkr}ifqw=;@Qi~A#!-Lb$m8kGZ@q<r5Y0vKZw1$PZF>=<~$tz9})r7)0za)pv^poOxrJL=ATk}6|Fluei-lkGqJ&3MTbiu?t^Z*s@SXHv#vora1b{4_Cwgd!@C+c%!%E>`2XJi5DG_Wi=U2Dp#h?mlKbhK?;-%EhH2hr6jS%DsEzfjI3MWYu3Or&HP8ik6n%yUFh>gq=^5G?bK{UOwubSX9sEA4X%0?+x_*Q+sy_;Q28U}e)K8Z@NFGI5q?%9`}(&2c|nUmypB(kft`$uQNT6VQqk9kpQQI#eAw>jIhH(}Xds)+8^llyNJfpM0?v`(tFiZ2RnNd}*1;Qf98#Z%!~K7@Q;$nkjqF27AX=D?AfH0?A^x5|<e(zzd)xt=O&-k6LA5b%VRAy;dg>MJ;yntwlnu5tx8gVn@5bnm%1BD-m9LJ3`bhGwkELvDfXM2C{we;Tq|V->Vod(O+N>p{Jil(Vp}Es>|BU(3Y!lLX7e#`n#@O?^>-L9V&Pm_EEpW^XOU=;dR7Gcrn~%<oxenOs{`2y{e_bE@}yM;e#Xc>@g^V2U)UMHsZjhyyjmG_0=dil@IzNozGL-HQ|SC9P?jc^6@?mr>}AqqFn)H5QMwCDwcQJJc8t5bbP$Lf3B&%UIOyuuI2hZV3eT|8llggg@3Os|8yCX3eKlx=EXFGXAxl%zWjxOk*EBQt7YR5Vj9cyx<%i87sXdVM<LS3h&C)`NAr;^U8%8=wRUBht<MWsl~|3OWQI2wYoMOJQyd_OCY;(3k>OA6OrAh|+~%`9UBSBJ66Z@6B(nGU7Kz!J`7C9*+Y+=dMXFbita@cg#-WBtDv$&JuJ-h;Gx3-dp`i0^_pD<d${S2XXhzZD>F6bL^y+SQW-<~Z<ce|G!Ks$`Vlr+vudZ{_c)?A*Km|B&Yq|rY&I0aP$F0!~Wz-Jfpk%7B5EAC&ETqj)tFNP$k1;BZG-r(`<`b6!(Pvuf0AOCb-q|kwVB&5Xu-cKo@;6&{y6Y$E;`hqVb<bKGw-#w|>`GV=O7CSxt9~hscUhmOqxMko&+S}iY9UdfDNtnjHt^(hREy;>NylRetJsLYx4_%3GIt9k-j142iX~8JFtqh59p5RNaihWF_#Td9)n#7wS8Uf^+}^p7M&-?{qXV0?+44RQH{EAM+fG+<S@-Co0yIl+w$pNRcyZ`W``${xa!FFpn0q0zs9UdXE|Y(Hp!rtU%RKu=T}F?8YjW&+9+a<fRQ|du)(aFWL^wRU)b%ImpypGBA#;&No~KKSo=^(|Ne@0hxLR((K1RxLS)wVBfc)%CRs<hWJ~9@~T(M^mNzZdJ?JW&|(*H?C-9BIClM9ZxqsR=3_meKhaQJP8$yHb)5rM>~cF$AZcilnbTbUR`SMP1y@+Ryx+yr*nu=M>@!MPtMlHL}|SUlpEvHEJI&Oa(7^b~lN<8IKLsfp5eZybR3A1j4(vDbS(P&+$2bbT)KX{Dtd@kLd;Gsv(VL{G`S{J5A_zyHTm=JbknX=QM0xCUBwg<yiBvWt2wP@JtiY!|*6!i8ghP!p28r!IQY1xbqz`omKrS>YLrA{@gr@{xhvB$k6D)Uhe|S3-wQL4irbbdhd{L&vE8_rtd;M>Op2$-MW&DWy%r;M`Qf`Z$G;LaU!>Hd*H|#Q3Gzxa!}a4vn%yA@>gVjOcX}9v{Qv6P5jrR3<wd^9mKzKkW7Ar5O=tok8OLridQWH%nR|egT=}%YqxeOxayU$u}q)HAbffdVez6#n|*pXFOp@W%lQqTqEJ8k+n8nd5TG_EM3&q2QX&Q2Mf&B_AjI8dlY9_<$Ok9Kk?%amcxDu7%uufovpK@)d~o!X$Qtl>AbDxw8pT+a4-N5eI(IT>_hK<>hP(eFrLikT6F^11{|TC%mW)5(QGK<4lJfXrn<)V+-r^(!g7MB0q;dWc$Vtcz^yBxN>KwnaRrTXe^d>+Gpfr|xG*&W4Q~*l?s4+x8Xf?=JAhpnMUp27ny27*WsvLMqK0e3K9fcX0m}{S+)6>}EYZ|3S*+(}rx(2+?(Lm+@hIqZ7mQ+7ePlq^VmlDfL2dZ4Ey1PMmfWA-Z9-+5Am8ix-S$kU7Bfa=I$A8QFatV}e;=4ur*0!HU~U>Q)gt4s18$**a@v3%8Lm14+|rxHso+VarX#Y7@KG<4{h;{OKxM5iN~E<a;ONRKVf<``B`3)j)qUSl7DoU;0-9Js^>r;%CY7{OL8=VkWW*g+D%bTJ6lq2x(oFf#addOiJX3DUsYK<20gu-<B|=FLU`Hp#+%h)%hKQVB-n@J!HUL12k-_3rgsi}9qRn~A{+BFv&1c5Tk9=K7{)h-&gvmg7ATb9>y`#t=G@OSk!5M>g(R8(=%<}mSAm{cng6lWV*zXmsh+tRyDx2LmcD~q4AjsC^e34RSnK)le0pU`ZMA)W2Jt#P|hJ;2Ud(~n<%fKT?MJeA(-uDjHLW4QDl6yc<p#S(_?;u3YVPsePmA|M0qT50vt(DfG`9we$Q@KFz1J$>=Fr+v#?M3Z}G7=fX?rwy3-ika_jy|<6f*@~D^c_zMIz38_f!mRw4`xh-B61*VPS1N$;CkCx#mWt$XET!bwFuqT#t@{RMmV0D_2<bcCq2}=43(_JsIv)e$a`lL;q``f!eu~lt|NWbT|IheMt9+_mlo_A>-HY?Z-*bp;~>~7y-E1cD|!+ByPlDDF)#6uqivVMMbDnS?MeUuEZ9-rkZ3+qu9w;Bnv8%B6qTi28UK2sf?>iDToGdvb85io`OCvsM;s$l1W@YAuKIBE$CUM1#rn4+%oWP;h;p$2b%0q*$oU4{)9?!O&0>=hjJer<x40gQOd9SNr75V)qBT2VGl}#Lh_8GuL{g36?ezQ9zAr}p&>1-zi!GxaJd^{AMx8p07)U^5pqA+>OSlc)(rmE2dlF&3J77z&c}d|bs&dz@RCQgTT`&5MteZb&-r)Fx{8>>7I3gS&<~vlvkWllAoSo0}^bAO;SWo=oJV5pD9j~FXNNs4RK1(xxHe6E6?BNGR#E!Nlz0{4e;Hqa@7^{RH>&|#RNjt{E+kVM0(<KgZ;+5N)Bx{hYxu-RBjUHjmJuN(cP{?7Ve)Oze9+{tE;Y@u8Mj`Xc+JcX2B{k7lO8_<%9bc+gp~Fobq?e36!Slgv$Rrd(G&`efDv)+XiUwb+`GIQ9OCiZ<3?C?YK?jhiIwalNu}RL^{yx^{2rf7A*bWF^w!nQR=OX++2byAampD3r1kaa<5nW)>cc_rL9F%{qD#B?4GQq>luRV8Q_IS@J5COW>#2z>tG0GF&IdoZ~7>hK;8~Y#WO>G1CC3>`X@MuU6kNvmT9Jt6;L;HhAdsb`w%WJ#H##c>!4-eqYwECvMLx&<X0w~^c2n?dW1{%Uvi@;C<(=lK+%!e-0HNEMy2=MX`4^Anqfmp!yhc*czby<GHHH|FC9URxRu7v8VCvfTqqPdN~D1KRBhc^P#OXHV$z*gCXLMoJS0xqL1B}4-(`lyl~_}`s85azL<HYWE2v_ersQ2#_%)M`y<k2yT27?v<lii=_P#w75D|NGt>91ll(ReshG_4O1u_K5UGH+-Uf<8PZsOf4jFg<vFmtdf>OAzoD5p;bg}o?-N)44v`J(5ydXfLmkRB~mClotk7ry==_3W-T?fkw1}AjW!`@7U}MhWx?)ll0|BnR<1hLneG-ZNhtiQKLSDgHqBtl%s@uw=T37FR%e~-0R8Qg6>{s6KWg2h0!yLpAYI%A{ZPHFlT+r*l^?&qV3+2>QzCwZ?UY$jg^q~0?rSbb2egal^X`F7*u%RlS?QxHU)*651azWtJ*d6`OxQ>1EWJxFha@7AG;k(2!{5LES8O;*tCRvMArsX930WW<!X#T(mv-PI(KQtX^jX|sn0{qmM!nGG`#YvSxJ+l)egpzchsJ<+>JdY9x&>lvCQQ3^2P|I~evcWNd3uF|mu0d*P4X#;PpTXVMVgM{EZO(t`HR<wFY$OT*JyC3SDF*`&hB~^u=SG9A#MP1gcG%F{XU9!COLG3Ok;7QqgCSccj-ykRk(i;z#>Vy{LE`ldl=+B(`7sRNg_PXo#n7(!qKUvpeGR(zQeseJGzlHCy_yO<BO`f=?zC{g}2c{K~N4?u0$vG`aNzyO1ye8f&7mCrAhU9eBD8MUZUPk-8igQyK1OdG1fsCTCvX|z_6Qx=;Ynui`V^=7q5<@H?Lp*Cui#u;l`vGV`%!tg=P5oO>}l<G!K{wTEdtDBTCUr30Enl(vwJhZ1MhS?UDsdDVXR4r_1H+&V&^mGQj>%@OF{vz^Io}?W$~#?9f=qqMa&9dag4i6~){ca98n!(W1bi1>ClZEWb&Pm?uI0CC$nzU0BkpS-!+eK{4`x@uXOsWF?ABW(_O-{Pd`&vwToUG-E<bYH>=L$jPuNaU;rbjlo*!-KgtLmx+`mcysZ;<Bb>3oNmcVM|Y0Uyh#z)Ij!BwjyV7w)H;wDp`OuBQ+pfqn)R25MebHFa6W3o3VSVgYr|0bcuT3q#;EXc839OLWQ5g?G^A%crq`5z$nmhGf@|U9s%4f|?NG2fE8`)_A%hRKo=D?!ZJm(MA!IZ$V=wW0-K|B|nCS#QZ-^;hJD~2olv8{g?>w(ITmq`4ts~m&qICwd_4m0cIzFwrFLv;NX5tkfi6V*achlu@Eq;9+6gS-_?iov0f6eY1B&&4HtrtN>&J*!!$Y_^oGbEu(xJG0M1Y}Al4ElVX&nlMmy*oPj?cM9-@V93#-Xte)em#19962&sny>0i#U)tYMHHF4+u3lgtdn83s&01e9CUlj!Uj2({Lit1WFaoLcZzp5jL$xkW|W?l7!6Hj?AtW1ZY|F#A7*zAw6)@Wr|sS;M4>LFL#x2rQyqZEk}*1!g%*;Oj81JT3swH$Rs0aHbPf-RR}srXl@0r?szOPq;&nFjDqd%^P{mbt;Z=l^P(|Xs-I^C^6&$6Bf9)-6wG34^C&;a?%R`N9HI8afkAU`IOA7oumuPK=aoyDX+G^=o3E<bB<@DRZ0_|~&e`qT<6<5jdHyx8@Q&(4pUM;QYRJx6*sIAQhxBiFLx_Cyt`dGA~4a=c*J6+P&49>XQ5frr5NB-7rfF*4$P<G|Ez><y2o78(U<(pgA=B0}_x2=!Zl()48G~hOaf(>h*)Y?elrWLI>RHbb#0spv-prEb3mO#3#u(++6a*Rp5>lJT$7P?zOb+2FC{&;K$@`DGi_PlHf#<{q0`y;d+Nc$ki%@5Uf!0ba4H$P_E0kaQN-2C8e2ke65EZEQ*Ud2{;@C<s$yjr%^*YBX)N*8bF%fYwu_jVV<8fU!eVc%{A>r;uhBLP~$)3fa^4u?0{)J8?lkm!E8Xp4p<R`#1x;Z_Ub4moZoX*N$DXqLeIp>+Gf*dXH1qMBNVdF_wzx_xjir>&d1#oOVNmWj2Ia@|djj)q!Swds~D0cd))9P*(Ch1HsSTSK$4a09U6*1#P4q5<0e_7j~>n}E@``v7WPHHLQIl|!g4t7iip-V12kD%w!DT1nDYB<XqSLR5Nf4S=NT&0v(eqYlYZf^Q3qNg5lH$7F154>#0po=vuC{i(@>o}_UomURuQ$+T{zWSYvtI$IWF3EqK1m1L5y4x+ay+KjH!#g%-yDONLhyYdeiJE$CdC7Vi1I|HK@L19%?6fwZ!7R64|P^x=&`aJzvk(}YLADX`iLD=|J^K9W7d~@7yY2#?q#s*CIh5ImcjQgDk1PqxaFL~f=!#Kof-7e=?S7myQ>0Js@(BqkicB3i06|;oSOY)JA&EEHuPaV?)zY9=kIm;F?pSgiZ8`-Oes2#uhkE3Xh4@dC3vSh8b`^nDqp~pR#U%{9$v?QuMDWqzTllmpal@;I`4_z|L=j2z~HeRfEE^+G3P|@AZ|J)0~>6$fCy<uB_HZ5&21{2oA$et<)l^K1{f-Sb9=P8<ZRew(Au_Xf}3}AU@mg5m^Yn1Vhe(Kn1GvOMJoQ|!gc}W(JC?(HP)(F!<e9^Vz0>PL@&r?__2}z;7UGmATIC`M`JX@^i93%}#-F2*xG3?1tBX{~<a;8<58j;tz1Sp>uupabBEDj~U;-=k!WNRmqtsP13Wh4A$mbo>R$A%PIUuyc?mNm;aNB<%V=r6E>HfF?nn%kKY-@|VBA7wjip0`al#NR}RnB^|JfXyk<2OMz(YO}`Z?M0E0FwV3h5h&d~YQ_tc&@L{bHO4ui#4`gc%c+TdXF1ns+z8(Z6G<f(Jz5QVq3ZU1ObBI(L8k8Yw7&DJKm=gxECr(gyOG`wqS#DJhj>4==Ug*B@sel8fzWoex3$oN+|5XL*sI06bC2AH*oAxxIj)!YsTa6_7%tVhxEMY)WTB=*soJ%z0INIO9`jWw_J=ydv!P?`YaeYFfjKmt(<WdbA8z_jHfndM$$yevdyPi@a{G3RsBPS;?X0}h+IFXAr{SD9HT~|=a8J;zv9p@Vw1T0ny{!T?`BnfeYjFAVpl`$WXmIU`rYsHo&~9$>a7CQovl%TNE#X(OV_>_VEo>L=XSYyan$J;P*KAugS#1S0IS8k_;R8iKRdC?|(r!z8zYQX<uTI1FjR=2B#Qzl@04t-#)yPPhDOW|X;<hiwm%JSJd_lt27|nM}EXU1pcyGm)K5e{l^_yofk#)*Ev?UD6j}{~81xiC6>b)<%I5qin%(EEGS>&~6`9`2Qi=4KA@FCRt!&WUxMiOdGI`8cuh#VyhMbzM1Ktl0J00yd!TUw`~4M;bweOl`--KHlL<+yE7>AS_;l$<9pKc%{Af#OPoEEvv*jS%YVBp0K#rO1=0)zC7k*4M;HDLkvX#k6qQI4r?<93MY<xo@&Ns5tGQtl=XGDEIa!pHu>YkI>D*-a{pnvimpUNCb^RD#jHjUs2pYfTkxAzt^9dYl<E+E#BMJPeaMWtge8gjAuCpbD*FwO}WAl_VxR{I<$t_e!TW3Sf*3NfSu0Zj{=x36st@lSjxOiXA3ed2NIfvEuBHq?=WlBNjfBr;SbQ47rS~%Vz}5y15pO}5&Qa-u(9RxuA>5#JbH+diARq$&9!&h9gLT2pf6zr%RB$fv_iK@&2TRcG5RH40fQRs0BG6<uu42lAtN&s#sF=MWk$4a#N{Hr%c@mAcGDhE+&8>8SErJvz`sVEsh}R9Gxl;9xQ|OO{FR@MhL0Z_KM>My{wVmop#tJf?^D_|&a|mIBGLx<3RU(3xg<0W41KH(p=(LMWwUI2HmadGPkTXjx3OdKCSWAsO`|-LF#=1bK7P^?(yJa|1x23|(k)|GGRovmUMgR?YY({~y?up4zcvxiFjC%v^Ht}I3op|n<<)hO7a_?euq~oLuac|?l*(!Vp@+$5fLK(MS`x2wK_3Q(tBW-{iM++XR<f+iK{}aW8~SAKK&Rg)@n?U0wX8a}aC@4rXVs{)ds)o0$Ga26z_+`<A08zsknpJGgDt(<g_8s=0c}?0E{c~E^a$G98)#s`?x9zrF8dgw1A`oddlMo4E)cfR7nw){hM|^&KT0@)O~Bdht3SzC&A^x03P7avdtQ{6x%~I=-3d;N&LqjScNto_%OUoX>&FScN*P*;H;!PH0cOkW-aA1;{xX{_M;)%ci}|Sg)q0U}M3gi#s~Zh)gQA#2bg<e`g@MKFg}nwFbiTECmnv2!pr6qqyTL=0&P#U6(UIgO?$s29(@)S(s9098=)iGDoO-IIlf0zJO9eWW4I&9dn3b#|jo=M9=iK%>*o&Fmpm7=1IqT!Yv-zN?>W%Q`6ajE6Sj4Krsx-oad(b~0ZwU=A&2$D#=I;4;6U47QNVp029pt*&1o3MQ@;Y7KbAvG~%{VK|Y_f~yIW1QUJ>BBp3v`-299Q`@X0Le5F&ohqQ(`V+?@uBHee;d7yatVLAqN|e;srU~O*yT%4)fJ*oIa<GDDBy+k@Rp&&d2mm%bswYD%g{f9Etnm^7_75l|dv5L{<(ENK=P&sd2>Q^F>Aus<%gg{+pW7d)!CLF5Gmu-_#QDNB;(MKCZ&O{?ODAJr-qsf(?hwqeZkiX&7l^quEL|AkNP;*~eto4UHI+_t78ouA%wfU~kye9IaPqq77{|AzW{YIesY^R|i8R;x#Ep?>p1w_xl~R+nWv_<G<4-{CjGhDEiO@^D>`IGR#0Szwa*3(z~V&e%#nw1~1}0&jdW5jl=tXzpdH$3uedd$<O}&;~js_M43|WYYWAD9}eJp+FS1p9yPW8u&L#Kb4%cxHnncN4ZsMsx7}}^F|bRYJ>OM!v8nA|<CM{$@3Z%b8nbe$PsWW+{XcxP4G7{SoAzug2xPX{1mbZ!h#4hOfmyUK$HNWFL0~o>g{`3AY`*jkrK|aR*=IFF!#kuRrea<#l~v|Cvm+K1NLPfsQlpqj{28V@#%SOYupoe8mCCDaEDu}gDJJC6)nCE{yn6BKg$kDu5BCR;qF+9AiNME+h)JW$Nb{tPs2bYYX+|pOfwrFbVJ`xPCEiokQ>C>u&BT?N@TPm+GHJ1?muwO^FE`f@c@H&>fFt!&Zk6C_@9#bQp`~wH6MO;AVJ5J>t4R)2CX%tSJ_R$NIA>2Hi>2j*L7W?`B=pd{Li1|bR|zu6kGFA&+J*^zD=4%;<S|PG&4JERM5(THLLutwe^CM<Cw)-!R;GParhV29O4|7d0ia3~Pc9}3UJ@HG@pkEqW)t2qJZ5mI&evq;bDm9svu)*nffLu^%}wG5$~^Z6mxcnAHgg34s*j4jjSu102Ul`NsGQ6fZUPJnz;S*TNP^KgMzWx5#cC~MBSBi9@s&h;7Io|aCDsr`K-umn3%uzpqPA_j0W+o?W`d*`e2kh&IExo}gCd18)ezW#u$kp4%D%)@Zg|{V=4wZ~)=>tO6|TwnN1*~zwCSAmi)mj*ee~wI+csAWZ+)OvaE)QhO;BtM>DzO|R&zk>(hgSL2HirqPonz<t=3_{k1Jppp%Ne~j!Ttc+!sm5LnRXB3Y+!w1t$i_D*c}@4@$P0!=hn(0jR9>U`?6dhR;<_>A_(fbCmubzI%eF$4nh~ButDvF6Ox2jc=qKZTGWw+r4!Rj_3RH(<8cUPbi9SI>U{zgBX`8406kcNL(+BU>uBi3>!9zR9={3I_*Wp3@&zdnO^6`dewZwd7yA?HV`}QlgBA?n!4*u7F^aNc4KRCNV5Lpf7MaRHVuufW}6AimQkgGO(@X?#yI9}!E4A_JiV~fKZ?<AbjvZ1=I}P8iw)yTmtXJHxB<>&d?sMr%OQq}>qXaj%3<(1YFf*&;n3M=t4cAR2nDia8v{DY>EpAdLc%>edVBIq4;Cnw6rnBPb>z7jT1y3j$Kk1j)7ag-h7=~OVF~}ObqV3^Sf%Q?VMa#hz)6x-XARvNXx-OFpgy3X!G!|kq}mKOp4w)bW!O3!dLM{7v4|Fx<$!nU=WMko*0Wjeh#EHx$k-Nz-|YQXYIKp$*K;-b^Q^kbFi$Q$WfF6+bOsZ}6eO^0d0F{v(*X^Oa@Ej^a)2m~_4MkGNxoXZVD#VVd5QnVNixl6S(0?Sy{N-M%~6)+#WnAATW0ITo_8XxlNO^Pp1M3}65Zasf!B`@kSGzhPN|ThSPZx1xo69?9;HTh1<3HQ|9<#ZCBX=Vw2SH9^)Z+{$*}Qex5TpxaH}$u7C~Iw=^)vb!eByHxwDh(V!*oqxj&=Wbm?LfZYgdSUUQfi&5r{$jSEzDT1zT)Wb=^&!lh4&gNW&}k=CC6Tokh`UA$RhaXLGs&t~@I4Ap{wC@wqAr!npz8=S7jmw5$j!MdHRh;32`7M7JIDuOO!W}NM+m|+^#jB-)uRYVE9Ti~jwmQUEDLE_0Hp{R717|Sy<8p)jU>i}DjwNMb(TfBhV^V#kSD$*%H-$+2img!2O3C@oonbSD)cXXbPuOdZ8<v>v`F8YRjBZ{i+*~1iG6ljs|8j=`5TU<ZfhTb9$56wxxKnSA)t}02;6<k3S1PHwYdsgG;+Vs}xs*l!~Z9LunqqawDj8PN=UNn++WwItEu;oF++X-)&&P&#IzGi)EI$!<z5<!kp+1SQ8l#;kMf!4u_TVrq?_u3lLxoT6hwdt+W>x`F5>}~GVJGK6z?&$&R3l45SNsTh8^tR7$*=MaFym^!GrD_zw-@rvW|GWd$3bHQ=0&RWl>lbg26V}EiVpL)hH+I^=!RhQHxnsja;nww^A;T0j62jn)1KVb3$I4No9c9J|8kn>s?{h4LBC_`QbDg7)jAOWk0F~7iS#=#a1M!D3nJn2=9zE>G=`7*j#spBAdO>3p1B|N#nn+{ZX107P2lV>^ffDO7%o|3<7}9sbEf&i`_ygO6&C>-2Ht@c|Qt=oDlw@Tzia0sTf#vK=S=^w|29dz~Vu3LgffP+7;chZdtND7CXgzG)xr8^cBWZM%|M-Z43b0!fGi_L`hv)bZJBeSg+P41O?BuK>^q_llE#%=;d{Cs)=%W|bhcEjlviOjT@3oKJK$j#IXDFPxl?p!f#+NW3c&cyHs#^3!snH4gx%j=KXr&e$H0mr0Yuq^LPQcHO^ZhkUorI1FkA)pCIBB1(0=JbF06td_(Eb2}8_IxK{wVN5gF&+_8OoGWx~1z2=}RS9`(%1iMI>3$Q-;!xog-&a`LlY@*n{vjW1v%}k47@n%4Cv_KDy`QQ}ofS82q6|{K+GkI+$sEG)%<(C)sB+BauIh9zA?U#_f~QqsPxeE5QQ~kJX@x(MR+96We}DIXl+qf=_IeR;tCMKR&5DhmxpdJiVYe0asZf_A*#R7f=ue*#}M#sbQ4v=Qy8dP%~D|fz_lsgs9J9=g9m{Wig#?XbLq#4V&+J+Y<3uIi5F68p9;0;dejLU^qIseB;q|^@4C+CMFOSJyVL2Y;w$n8e9(z{Qw$fr%lU|jmB}jBgXFK<$5}uWsG_Vy-t>VWVNF^1LTkk*d?sYOZvkcU)`r0zJRyf+i_Dz_3chdoSv;}{K2heKI1IK9ErQK3QVmfG@WO2bnS8pPgK;=OI<ZW!E0=p<E{!oyJ=}tu;wUb`aB)4&*7-b4)%y6jyZ{-ys>6`9XtE3yA51L!w+qP)o;&U_@UWe4qoA`6@2v;Ma~$J78phZpp(_@SVC;su4bI0nhi=v(NM?^WLA^0pAl1H_2EZ5&8(@+PcilvTXivUC_ZpwqV0j5S+3z8iluFWG6(xQvMA;qo4q<beIE!Q*76YL6EJ#JN|{y0VWLe~ABHMsc*uCG2>KY@=$cYdl=%z=0*+y?G2J5rcL2REGds3F;BXQPWP^)AR5u0a4x$sKn!3D0eXiID!?NF*89aUY!bCTK_fRN1y|nCsgr%MXlJUUkunkqhHdGxgZ0aZzr;c*n#@`b4k}N0`Q4!+dIzAzQjV+>ryGleH+p%BeFg)c@X~rjM0y^DVSaX_~tVY1o)`=Pi_fTrSzY;fT6$ff#E_Z|e%o;N8Fn?P?WN764oqzYNWTUXw(iDH1FKY2eju(B;O||E4$jFXQYJcbjC`-GEUup-?(K|Qt=|g*lmcy<5(eubTyFa_6SBs5dua{e9>dG15O7e#wWaBELUu%PLf<f^`;M9xO3vD}~qJa@AB(wVj?Gj=Q3Z3XC=pfm15P;<X@8C(K3%uJ&pD<!!7lq1>hkOHh(Lg>nlaSowgz?5l3&Vbj{#|?mKDwj&mP)_vehSFjfNwzT%Fu{WjW#kN5ucqpa}hOC?cHG4m;~Ca=;)*M$EWDO)E)xugs-jq1+y?)DxXY2_fsgFs(Q7YIG&^@UQFohw1gvqwo3m&oL$L}nys1ecGEg!cCA3Wx;JR!5!pnEN`hKPmGZIH<0qv-VNp3LOuC)6Xh|%scn+`P>!Ny&W-SdVAEu#fx<JahL&C+LA{11>1jZLKW9zKei-Tx+S6vp3nRXj?9q29#CCF3o?Acot`%m)1=76u?8z~9rgdtG?d*#~#6d*n($$Eh{JqbPIXhKoih@x1UoY3UvGdP^2^S^PQ1JecrZOm4JU{{J3O31W}OYIUN)L<Y)FiL{k`Eu~rRXRDQ;#j4&)k$pm?GOP7$ydnWTtrxIidzNqoCGMt$NJW|0LVa{iP>0Eg16SE4h;_?RO!umh^iz(99_zWn~=|!#j1*^-i|IMU{p3hTa{P@gCheoks_d(Ve#z`aJCXG!*G;~X?`)fPFH-zF$_w+8o@&li@2vNgTI&+=O~E+)*CwB)6S8Me1TkH#=yd65@U+_VG8Wz`b_#PQ#i?sQ;lR@1Td!mcHMLCxPJPP_iP@pr^4;Y)4|wx*Ja>)ErIo-$sH_UK28?nVpjUjn-rE29wXyM%I|2lEcM!{_0y5}aI6PGkrYN9;nAzGhYF1|$e%_H^#WN4odA9J3(_aooFe9Mh|`jDPc2~6yd_XsPI{KR^|+V=dXZ>AZa7rwkqKJ&ih4<@!@+3+w=Y086TTSGKNR%2J3wt65SWudrX{JpUeJ#uhMCtZ;VL9(Y^o8-Vlh}(`3xw5S-wp4_aul2Ic7W7!P<>uz+DdauLilK5VPGak{D5TkCtiwgJ*>2ho<A@+EH%a8az3AbF>Rx=10H|d0y+v65iJQ9qw?pvU<YN(R?_oYx#cflxw(+ZB+PRWD&gPme-1HKXU23?nz!lAndrU|0KfpN<xD_u)MyvB&7hu5C&ZqWi`!j<BwMd7<}RSVDNDIsna$_&6LKRET*UP!1OpFN)OYubU$@NK{(*4YDDkmIRoPHBK|mKG3V~bt52N+7)r~SkXrsI)tLo3;HCQ1qg@zT`{W~MD+t6DcEdp*XzzL$fza;OEWhfw5txLbkb>Cx+#HdD1&PHP3)AZ?oEeG&W{@H^kr!1x0{n*kYg4t6{o+vY5n<&<_WNwV%=TDX=n|ILEUPaJJ8gq<f^t|Hhb9z@*>pCr7}<cu70ZiYUkzDPuj3q<z<2BPKrQPpx9%q|`VML2Fv1Crw&#RwHqyF!<Yscn?>Nz{96Da&{8oUwSF2nHVz7>miZrfNm@HBaoEP_SIj5)A1`p51BV%XGoY`0FQ{YsoVfqO!?S>6O(~a;r7&u)owo(Cy?%7wANeKTgoO_*xdk}RZ2?8WoEdi`H3Zh$hx`2^)5r#rH-EJ6kdxyAKo4T`By`i%jxwQ$G0}=-g@ea2MPk>O^yL$RHq#m5|r_eKEjW+y-SbZy!=p!ve@HGicKJa|WyJUfGdtA1>wJqxi{h_44yp*bAG#6Pf$xH$MADFQnXq^!ImLx>(CUl&WL`Yt?li2y+7gr#n",
    "softvq_continuous_online_train_v7": "c-rlKX?Gh(lHhm#iX7;5yUGGUfR@~D%J6GhqHK+IjYzw9M~_=90tK>6pbD-kfTT3)-@b9=T?ar(-8;K)`^{KHR%YZC85t27$Gv~sUzf%Hd7AAf*+#5xtIIsw>-Bn@zZ{Cq0}*Tv#C>tTNV6;WH(n)0wJs8|$jdSmZ(jU*ERvg5UP9U6JdI1SOsn)FuF^ak?(IEI<BKdW(^6EI@Sjaml<+~sRTWRK%0nnk#N)U7C(p!1yb=NY`{K7F@sESyL$O?nO$i^xKmIy=cprbp)gFu|Sv@>>7z%)<`W4jgU_j88`7BxB*Q$uqESZV%X!IvhCGqq!DF*Yrm_ph4x<Y7aHcyHqn<gTGk==@U5igUkN^!O-l1;SCaS~@D%~ntyAf{C*UjO}-z^`e(E=9hs;8S13d$Y8xiWI=&?9xhP2vC&qHMHiDq}jz#h@+y&i&9MU<tkriGZ=A^uP-if<a=kZYKiz-1PC9N8s<^O;hC5x$?P1KORVDR64vzOI@cfvWjag5^fJyalCm%U^XTns==&O~;#4YFZV{BZI2#O#q+Bl(fL7$NlG*Lv=3syGU;x{i#1{x6nO?2(G=u3RLl`R0t=vD(=hbij5p514LV<l#F-lhX^fLNxG#Wxl)!$p>@vMZ+s0PVRJcSj83C;ojpju~186NKKjfb>&172E@<!PC~3jLl;>30C*r0?`$P*$_E@KF2|ugfxxRduL`t0jcLsSUf<7chCmidixRyePyXiHi(yBNyk1DDv}lS-}qkmM#Ia=P(cfh+5@<4QW<F4>OU^MUgCSVYLNsE^O^NAoy~)cQ6zs3}LojK>uaDTrIE`PNOVgCi(eUU&tSESth0a3A?iz(E61yMeIMhfjxyO4lWAl6Q*<>7c;@lVA1l~8qicD_*WVeXDO{6Z2t9SI=w_tMY3ANvi46W-nzuFFQS000W?_Q;vu4la(!MV6)XyL%nf1V;srtruG8w0MitU<5Q|~mMLe6O2r^z!vBv3Al*wW~xJa_30EC|D^~^KcHCWUY>_-Bfi{NE?8Q(7QO)UN_UdBZQYc38(<G(_!+eJcXyjZ|_Zdm2P0HI|$boa+P$*L3vQsj#TATAt%WR}cm_hG-vD!whnJT4g}pFVvvDC2n&!dW`Pa}T5m;3W${u7Fmz&*ZKzUY@)GAgg4$UciQL6dn=vfrAH~PNCO|acpz2$26ya#akq>PRAAW7cXGq;`|mj5fL30D%)UFKqjDBvmEA73D`$KtOFz$w0T3Mz<YaiTBB&bMw%N%h^2Xf(||TaW6QlgRaRWUa+Qhty##8Z{?1GF7w<g7MS8MG&lS)&*nmD%w<|z8S$qutE|OP(CFKeRru)m*%hjy_!kevB5p47H()lsegb}t5>f%qgLY5JX%r70z*V$C!7gTsAN2b1N%#h8Vntq<fRs2SMhNiIJc4MH?i*h)FO1drl{Q_uLQuJ*&$B1cIc1(vu!uVshRPFF@xkXbrHUx9HOqXdiU%+xjb{{XH_!*XBXI2}+w{i%mA1|&VM6Ri7^7!ouls&<sJ^bG+nFIb+fI`u1whEYRi0iyq%%U=VpNI)yPZi|l5bp7`fcy6%se<0$U%!3vG&+9%pGUpE=#6_}NbS<O-Sqp37&AQl-wf`zDZ-O`NI!$#bUllE*pQ3y=P)g!coV0K_<WHBAs?dNbhYk<8d{M6@@C>$4Bz@<ME^JL%TB1)B8OuX0cu{==J2lf;`NiqFQT`PU;WZMWuPq;?JCv?Ve29}1Je)JLTntuJz7O@Hm6BhMi+1$uL8VnlWaEWWz*?G&P2}Qq*%iw5JcY^cHdgdzPoC3TDHRvrt(LC$h$d(>nK~VXgiY(mvD9{&hvaB_ZF5~78Bv(;g%7O7^ofeHB(riXERFxfAd>!LaIY(yaWmr;4ltx0YfqPfqFc&C&>#u5hK|QZW*|JB48+rbXo@T-$Q|S&pYU^FXnLfSEp)I51l2j7sqt!w>{j7m2z;hcPhS0fUr-1%olK8)wp=SHt}Melp2StB!gPPJ6Y%5)hXVw;tB~aLeXQNs6o$=3Qo8+4e^@;3u}P7M7%5k_tY2pYC@mhQt|86@c7r4&RF&**5vH<{s5anT?Ct9S|!UM40&{YvB3epPgemBvM(WpAKi@s&aRZh_&>BM+<C*E$N}@Aq&=u@30L`&JHn8=cI3_r&mi@}^J!iHQLlmuuHpr)Arl7qXVdz^zj{fj8LsB2wwfa<IFC!F!B}s9-^U$h7}fVbwV)OOxi5k-c1h>@d)bmm^+-_$W)B}eh^jn#_-(*H70R6O<Z%Xefu0K29+sCN9ljfkPsP7Y#KQ;mlS|d);8!qoE_gQ_4*TNp;pmh;{T^RO`6|i0?+-_Z568NOMxrU~-8<Dpk>2-7k(aPLS4p5gg<ane)zal;FlOY)K*I5RA0Iw^aB45C)jdq6kAvz{OADA>+W3lj0lbN(^NRpDwUQRoW370>%6tvt3?89(dI3M^zwm#(SC3yF^$taEwvHFbH}v}M_ot6f9!GCqzdnIV*x683jEX$3yvi@1zkKdgCJv^d;_;go&rgn_)`uSPxi|v;1NaI!#P~bUXZWjvTZlg6zaN`A{IbAKj$TG@-X1-B{y(syzZ@@)Uj20Pb9D0h#S!obPjKlDOce`a%QDBX9_Q1oi_Fz_T4GuCb5<JuNk3z)aRl^(W~QNHG5jeD6=<pqe`9g`1|0wT3^oA%e*63lwrW_jFR>HrYNwr8S3B<nRQ+$U6F9TlPH<suC)V{IK5=pHiC?qy#qj`z4G|nC@p1u-<MXTn@tp)P`~cS?3NH`8C8b+|ij^es!0cb<%jDbr*&1lt&Audf`|AwEt$l7K<)Uo$eu$!DHtU42N{SN1Kl*!_mzSyh@2W^p*8n0~M8IkD0AMr9P?fD1!+>@{7}Mov;LDezk-Grf2{Gj523MlFfIeifT|*RzL4+X22n&!gqNR`(hvjMkY*4S?3y~SH+kE89o@E#dFd#<2cP#=f5u(+zVH^NliGm2?eeckjl?QaVUIE4>!8=W@HMKZ3qA2n_-C*L`ERgJ~brF}vlt&8x!dTT899}Vf$GCQiUp#OQ$Kc)efgZ<`7th~Byh#&U*y$o&MT;c6s4mSu!Tt8z_FBg$|MSJsF4f*Xe|p3faIa>G`S$Yl$@AB*qGxZ8CkGEr@#9~gK7Sn@KYsZJI(Yl|<Y+Sfb~GAU?GR_59X&qz_3aV09v=*?_W%C)@6preFDGMDBoXuF^W&F~PoDf7Jvsi3hs48$Q&ukHYI+$>%Z+JuPrS(Ut3iB;%2}ld7Zo&2D%`Ci0e%<{D4^LgMjah6Yaq{;A~-wyI-<XatLp5`f{1GLX<TSc>x>n4$to@qKI{q>w=aR-ghLJB?Ga4+Rm4kl^7@yfSI2T9Fj5+5M0(az{2;6qJAVD^+b2iI6NxSSA=DxUv?7HNF^tV(9I=s>qKJ_jm`qrQR(z8ijGOWBff>zj(UaFtkEoat2LMLE%Gx#F&_2ET_2u#8-MBBtqrNy8A#;v@k4{b1lj!Nu8;ycEv}l&BY!o~}FwxIPkDngfwQ#XYyVlRopFTZ$<qiv~rL$R*xx+g?d3<t=L;P7Gc3hPaYB|-)(&CVgZDOxl01MU^qzEe4=gS;5%@fjB$tG8;N=qELFM%}@diww-9}XD^g?M33NC5^CV3J;3!U*zgaXUf9ku?AU6X6JPji*(BVBYme9oPdA0uSZ1s<ldh-Us2Z$^*IonuxBD{BR}n_+zVwR>LCsW1SWWDuv+WOjsjlQ>dxZHBJ|((SH)J%Xq;LKu>Z&{%oDE%M*|yOXzW#6hW2^U(#yHeh_aX(oaQ^!8d#lFuN45XK9WbM`AGy<Kfu4jacY%K3gcNP*Q#S#%gIWq7m>SzJ&!*pT4!)A;PKh?@alu$XEHgk_rO&HX2$-cNyOxQB~~@Mh~o}!0kl1&gC>-Bn|+p5=x)VmA)~RP@rt7Wzo@9k>*8o9T&^>N)j41u(35}pBSk+Ruv*{(FPFtym@Mi82HcS${vYD<ENMF?8+a3JbVTd%QdiefW;b9LI+ib1_!hhjz8Q=3KgJ=`qQo8Yy#ABe!Eqjt$?b|Z?~$uAyC&Xa4R?Li_}{;d<m`V=tm$WSA!{?KRz({YE&gR6+fACQIe@gO}O{)EtDUy@pV11c<X!OJe|S0m#-;{TY)cnX-4naXJ_`Fot??|z;hgc2vF<HqkvZYPAWk{33OR3Qs|#5mmxj3@#(&VRz;D*bk{3<9t)_G!8jCh73fWs-xno7e&UidgG-Gr+Z2d5J{?YTd>iCfxI7d0S^u`U9Zd5}e8I1Dzk?s_TYU)}I#f^!D!5Ky@%Z}FtH-Z{OmAV9a?t}C=m6x6d6n2}X+5Tj>wLYK3F_h+pQfiUrwZnf;T{+9fLC#v7C7<(sN)Lu1^T^3;{yXyK9vH*aOj?)$rvAVnI$P{=kp@Hz?WG)V1*@ga}2#RVu8Y=6yF5nKH;DJCA2Z`*aNFxm^TTL56N`}FHhctB1ans<od+%(aXoLPM$vz%=--YY~d1hZDKPP0X<Oj6Fe|jh<*8WCVu-52BddzAYz!!#xQj-ac8~wbyq;2_#PYIlC4BM8{mGzdclSk6}|D})(V&gu;i%!gXJ?1<`z8J3t|#J?BY}1SEZPzMOhKdbC@_&MqG6ekU<m_&z}E}qo;kby3Df#5Ctb`gT&ft$><vVJ>I-{IY3o(x&Y}QqzSQI0*Sg}m97%B#1ML~<Lvf2zO@(f?98X%XJ>FQ))~x|HbL%lfNZz;`8}UUA`6#*HShC`p585pFxLYyM@s>XBHxX~l1`^Setd48;e#c?l6Ya7*|Wyeb@E0q2c;I95jO+kC4OPPT(aXxhF~S-=myhEGT5=;0nJ($WT4Zh2QW5&pfyFV4oS$YwD&cc4I~{Y7ddrI<~0_`X_1%ZfG8xt!2jS(mB)lSVzh4nBm;_r2Nss#aV509kT+)}WpMsy&g=lE)A_B3b!d)LHz)9pZB1z4JHs#bZq0R>M7j#e*iE2*kdF(xnRVmKeykF&3wilNJ$^Kizr9*~kkQ%!ug7fo0!@sHK=XZli&gVGSS;%eAB_I`t=*<i_FguB?sE<`5Ee&q-+|ORl2u+t$?}|`94*h2*$fC$AlbgIhL<sHL_^bq=0?0M#GP^ri@!eNE9wP)3>>w^EFyY8X%d5oZ^S*Hc(N2fl`SJQD3b<8769?Jc;MA#4tOH}bw8qRa3jT?i)69(RBY!Yw-XH|A;udXOt0}{=nDe>e_{NEvCf!cVOo|auyCu7_B#A$IpKer&(1cO_P%Tf$|Jy8$@7SnUct|x^#I8Z?$qXeB#E9m^7biL0z%lto|~4(+K|k}sV1O6N3?##KG%J}<xfX1ehuoy4Wl$fDs0_%vI*^)EFkNF1IlfWg3712Mg!GPT`zzPYi1E*JXK$4<-TcOFk%K3zfPlmqqHM|^&pIV@z|NU%@iMHvq3%|ux!GNAfGKcenASh@DO-SU;?g@kwlKxNNUVBS`guq5GV@(E3Gw{@%my3%c(>hW1PG~%SGlRfGx&G1T_Ty&SON__}Lh_mPu6sBL;uPz<{lxtHu6h0x}C+Z;5T-dI{3hdJ2L?1tYP5%7r6)HC(8{irFF>05C-_63O2AjO;QlfM;G8*#M^L@iz(It;uZrhFJLFN+~&TizTXd*)Tmpi%L@E9C8V8&`O#dq?wYuD%`1}Ot^k&>y%M;CG)ys3EM%SU!e4L0Y?U{R@kEZ`K#wA&mW&09iufXYTV{Uk_?J<MmzZUtpzZ^WdW&OTgfhg2}IGb6usmYWndUn?~$@%MkU^qZH!S0Wp$f0sgY8=oYw-`UeZpNIHNREhQ4GsE_p-b+0&at#d^Zrw6U+PXGn?+WY|S2myz_50{FV~WI-0aiL@6NWG7yTXD=SVJQ6VcMLI)tJWoh?vp^;{tK#7(?3=0A*mhaunQy}cs9%_njn(pWumHfX4{V>jh%eBX*&GwJPO(hAr;EH~q1hfCF8q0dLQG=7l%xSbeEv_wanwrjf=D#IN|F`28z5RZ$HN%I1Jv6p85pBrkjv7AF@K){@PKutxP{l)BLRG5fg|B#FrcmntUVSZ9OkV+0X547YHrg6&5IB+u4D-#f+POZn_p3KUIUifyYckt8<C`>1EG~fU`v57SpW|}%%XLML@P2nV`4mp9vQV)V>iCmYW8V2h7A_YpB{Yr4MEL;o-|$)_XO8>u{uzXKU^A7>2P)X2yMF;{60h~eEYiv6P6e?5jh+`b%YGEz)eA0O;i}k9onipM{5DtjwIBAI7{ae)0wuBj|Gs5cuuH*kB@Zp`tG$Lw98I_-$T>`B{9fI;=le&_7<U$7!=$m2;y(5m@$xQ4t+M_0#G+8qVqKlD(G2&dXA-4@BpvpaJS~%VI2TY$_9e(t~HVO&@(3Z*;Ol`VgmIx<z<X}QUT4r5{_z3=^@5U1|N~W!vi{B7j&4EMkOr`odc+nvX%L~5{Wt?c)v<50J#l$ASI};N=r?k{eoig)7xL!Wg@V(2&CDH#qWqs9z5j(VcqV(2>S{Iydo81i4=1l>&Zx{xIimQC1DuzY8COa`Dc0C@rBz!pVqJ$aFI#&4CebJSuBEWQo7R`=iA2o`&y3BTdyN~pKJvVF?0B|Mtq#DB$_8N9;m+h+PtHJFi+oI!6uE#mn6dGL=@SkqOK5XSbS8q&>R|Rus_H4CA%k{cp5P6esySH)XcF8*kV%S|04DC1M0+s$hx+G5GTQ70AwUVWt?>e?(0?2WubJ>nZ`aqWN^gae}4YtXS0^_JpmY~^$Hjh^4QtXP3ZV(O_P)~$FV;~z9#WxF5{a)3`|RW0VCYNKw%GZ#jjsq<_k><n)g9|Y|`;G>fE;AFF^cbXK9xw^D@rX$WX2qRXU*C)of=lrcJvQH?V2EGw5kWLqV?a)$0=v8b0D8Bn0$W!8s)T{PEK_kC|s%#On+$u#))Rivq4Fs0Y2QEOtY#G|>fHWeYqOf`!%o@#_;-fWR<E2M=z>qmlTBfIqKvv(lezax+a5spft1>(j^FJDFaK;M<1}zj;VT5WO@bLp-F%^YmiP4=NrkpnPkTv3jr&Zx+Z8(P&U}8&|NLB?v-r3el--jZYQWNedE?>Ukm_v}|Zi1-UalgXUMC*L?G7&F_gf?4ibM&98{$dSa12=^-eI1{^fFfO4C=Z(goX9MH=c`0iMW+=iHtJbQV&!k4#_6z}TsC{SL%@kNYup@BMUWxZ@eWT;Mqkt_+cY@1rI3-W64Akq@}LXPE1Nd`%*2#^o7K+85s*|1dZ{m{)c@4`5H%y9?b{0Y!Ju8v>xrO^wz9BGRX=L6QYi~;ZlT#`{8EB{DBvh7wR!<Ux0pD|E~VA(e#^(x9)V28t-c#{H;xz1A5U`D}m$U_KgEzpHwNKWLy&!IQwE1+_9z&3DpBM7*ef#`kqbau4?Y^lB-w$kVA1X)*h;ocJ`=$lV3dPYVO*Rk=Tx7bIJIN+jvV{CJW=;m5fo+c_wu33$dF(>fi#Y+^3QuIVXv7MgS@Zce9JJP}oDK(0XqxX=!vp(6R(;+Ph4f#9)nhkt)G3=bYdRp3PrL>|Gy;)jY!0ZqstHwsS#(gmu4<Fu9ZlTp=`4H|Q>(L!`LOR9(J?$1~m7&y!De>>;3y*tjT#LAQDYn)3@#aURG{X1xGF_l_5zx+BOdfIp1q!UeBENt;d+-3R{1sU+eYekSiK~WNXL76ejrPRVv?O#(YXP{F#I_p&EEUF1IAC-acR>QLmBfMv!;z$ie?!?aoi3B=GM^a=$*=0sGA^$IHhz^}WUWS@bw{k4a}r?7TIy7Ro|O`CU#NdO-0R2*6&;!43)Eds#@)jwLznhRkhRIDO{mxFImS>3Tm-11g3EE1qGeKkoy^deUmGe*2ACNb(t~q&kE`Ss`TJ6tHz8HK!E^KdL|LPNG%-wuXx(It`5Y@J5%54lv0AK4@in$1nc-_?GvydtDN`$(4WW9-ic8z8?<eE4GoWP)U_U9&2eCJ7-y^JP5A8*<X$nE{{4)Q^$YADjr43`_0;7-wjQ~Sb!SOhSt<g5yfI`YDFMUjg4&LEy+!u$Z;s^S9*r)sXRJ8u@9&I&4XW9u@79#}}TC53(sz9J3x>V={_a$G5OqlEc*r)qOTYYvQEkh_0w3IN=;Qn19XP2UD%_0F3hPqSq6&yQPgl;VzD#xedd(*YE!z<zuO#`@fBi5SvKt`u#_+rq|N558faO&O+_N*-LM_t^<;hffHBsWr5!~Hu{LpiN)1zgl}Bk;<uZ-e+R!3gpB`vi-v`am*J=lb9t0&4LVb@|JLV)xWp@ZRPx%!n8>KA|kFHTXwRG+&oec)@5ZRa&j9MD*(LVrGtzW0H!dVXuYnM)15Ls-s<zOt##PVv1)`d`6uu-er-9)9+bR0&a#?FchjOhc&sxm=`Dw&|>QD8divbF}fs#Fq04dd_Ed1n0!!weXPR@8_tu0Wloa|-8)Nu*L|<VKSyTMtWzWlyU&oU)Ov!*1X(_I%QIBO*E@M}8>j3B<LR%}9TI1k0YEe!+o5gFt{S`M)Xk0AojY_4-Bo+f#<~Rnd1QypoX-}w><&&w5+Mqk5g|t(m_P5bkZ3?-TJn%bhrkhX2f^3oqePR#0iw>sk-wxql=0!gslRM~orA+~w)XVRmbEK77LS%nTCY!`lN5dLq=YXK&!NF>F(`=3(1rM;n_9P8)w^L7YI7ovXKV&#1)%Xb9v+18tjJ5BzcDa&0U(DFAfSC|HrY0dFdR-7@p2WQr->MY%rM-f$+b6ddkZKSRuZAHVm_avS7pQEyOAd(|M-?8fF1pI^h(6%WxjyRXaI2d#h!uj^M;BvhtyrD8WyW#j8@RTUP+_qb;$3OOxOd<4N=)Bps=R^8paQ}VpDZElopr3Ap;M0b|&pI;iqqW&juRjiB?)zt^oBJ^7{skRI2;*Q{Y8hoa4Qut!{8Ql1(-&_`jzhT@-Ym<AWc`>>4=TZKd^|z;ZdeAIP<ZYws#q8MO)RgfiIHkl6A7g$%-=Ra~Uit@s``l<Z6~B3nwM2*^>=vv=UGCL_m|sJaY?o&bR=uZ~OJiAtiJp8&R^*K$vWXbh+Y{}OISg@$%ts%`6u9r<VH0B!)bOU6@(0nG5%sINhRXo0q6a7^H+z#)>6T&#0~BcuguCQj!3e43D|BeiTN6BcQ?<W|@ixHgQ-g!YvOq1fSnXc)7muzk{DhcUyWTY-%JV&)fGJ$39|jegm65fj6AEP~zljGhBVOa96ma8C?A`+wrzo~?1Rt-aV*$uc+653VXw;Lu;YFFh>nt!vwYNKC`?c8Ju*_QWI$6UU#FoK)}PL~4xbjOE2}ny+r7fWW`&kuAn4x{CVs!e$WMlM6&zWYoY&ZJ6Svpf(zr-0J0EH5eJHvBKYtGiD3;P}l%^2qZfQhvgbhvOLh}yI>t^SzTMV@Os}NaFM(TU-F}k<SHMWVr6NCql4YzNAmAiF;yLNI9e;0s7emd4{*d`t1N#<?WkdG4;(5UY?5X;{~k0?`m1C*_yPV}VQ2$p&l)zwU)|XdUzP*OybaOvC#Z39yjgsZI+hVVsDLZKw0%Fn?OxkAI>0E~7qWrhyC3`F<n&OC*;JP9NuEZ@R(o)ZFY;@PBjAqV(!bF+@>iuYsMNa@l-Sb@ZtDWwk79IWnv*pux=p(q6tBFM*LcKJw;U13+#+5LxjR}uxN7ib#y{yu4mpMyvu_cw6~Q<ed|T%Nq1W%_D#E3<M7`xA&n}$DGPxV5D|Zx9m;P>8o`|(i>b|?nL15a-N_viU$qz3Bo3yGoP7aF9D=tNPm-kQ)>+~49F3}7^$8DJ*qk|5dc$Vm$mH1fk<I%I%Z;vELm8{$yleXDurZ&!U_6aP}#ab4<bVhy&<#kA_(|F4n4St}z1VZ%3I!WFq!8ZAko;n!>S&;?j<$DzyE}3IQoxHf+#%17RbB$1XA(6<<B<USGwsO!}U3KeoAYOEsG6lX+!<nxai-6#S_K01mh_mfbKh^~lS`ekxLj>~?mez`67h<eE?O@*DYvzAyYguZGa2Y|f6V+d%tEESqKn_SLIQhFDPto0v;%M`yWqiwK*~cf3-=4sq99-GD&{aTiq%~wT@xuhjmKh}~_*qhoFY-rd#l-t3JE)s`UtgvRri`S&W;tM!lgY&xr70kjGc-Jxk_pNa(zCWt{QNw<LGogsAiSqE+fk+$%RHU&qnx^zKtx3+VcY<aG?T#t&*;)Dva>HRF2m0hq{PMede~nG1P}l12*D#;)98M{cmjq<<Ks?*`!Hb)c_*>;6xY|Rt}!M+Yi>0Vx?BR;P97H^1lS$_SNz%ZV6jAQ19EnZFZ|DH9ByY=vmwx++a~P$K<<mWt`9-gF9-I{lFt>L0X7=+;*Q$5U9oOf(~**F*}|5V+yD~j+kuZ&ji8~b>7o0CKCxpTD4#TFAj2T6T^k+|AWvaLuAfis2f?#5H7YoUcEwYWkcd{GsdcGNC_fQO{~qn9!`=mE38fc1!zRfIfXm41$|squsnjU3tVr0A82ORD?9Z%gvzw1}WU2B2juQReUmsIYf9t*BJh{!Ku^qLsFasTcO2<Z0I;%i4dS^h79(<^G*(2+%hmCujhcbg=W-*&yC7h=Cu%j@~e!Xb|p|5BS#E9@_n++z~@P@vm9m{Az3YKwDHqT@Zom1$d9)(Dp+^Qr2bakHl^wSEuzM<{7<!MhEBK9IM)?xm;3w>#Om!Uedo9*h!j)n0LEvgI%<_{c2OUm&|lU$pd{SeZ+fX(())}Gh|R5fIgiF)sQPGcyFs@76x+}GeNww>q0$WymZFw0bnl+le#lHg3t4DP$W>*}$nnbS1w>wdg)fk_ciUQO^nVRu6k?vx#4^NZ`Hb$g(xytHNP5-qwZv+q~IR>*9kX?ZXJ2$J*q;upMar}6ozMUw0BByj2b+T44Lm{TytV($s|JW!q)WY2-I8I1L@YI=1irwrxv5nK(dJJ^~BTJ0kVP<|LXrN}Q%SO+vQR1O)+1*6$ugLrghB%u$1+z38cF0-NK9p!(acN{JHus-tZKxJH@7ClB5uZ(c?6wUC|F}}Rw=!-}@8qL=|3rh*?${rQWm3f>H>D;>?zdli0-qe)FF^pGeP#7q1p-=oU+0YoMU*nkSr0||Nx<O-IkX?x;NRAor%_-iiB+#x`__tEpw~kX;qLHfrzSqxJ-yDd5YECd5mU1<u@@c+8$AmK8NOuvm;lU6B4KQE7e)$z#>*B$KpMFG_1~lB42JvVQ?|I#D5XyO_eLkKI<)HMkDY;h3XR2I1rv0tSdqmo@)%)_Xt;n*#3_!B2EzgD}W^@YdEmp7C{D|eCeT$z&-BriXE~_@H8yrm)$lbn3i4)-A+=aSX-Z8TVf0)WU`zGLwj1Pnk7Z#nPOgUIcW)9@^ig|`;0)qyJ8Q;<!sFX~b<Ip&=J;PU_np2H-!@<Ah8DJ8<%x6evX|#p_a3Ye$B3_l+vOa?@(KQv7>^<oz)8nDB*_yNm1%@{@?CX2wI-5$~T+&IC-pMf(-GMk>;Z+!+8dYzlM<4xDe@Ii7r`x@J^m#^KF<4eU({>c&d(H(JJ`(5i@wd8JJg=mS5SiAHLw9^LDA(tejFXI8j25dyN8m1<b8BWIMQuB-jug55Gi}Y@dt9EA+Z*@Nb<&wCtM&NVP&@Vynh>D-O%6^HfPRM_%U!WA%S<yR$}h&Kx9OUoGhvbg?C(PpUF8p}F}uwKu3MC)g=-=u*(E>bTQ|s+1@cOQ#4$8Ar`aG305F7;xm;ICzzJ%;>0>aA2L|rOg1h;Ga7)YtWEgpNMGF1^ASi2OCAtX*h!2B<kX1Al(Az(XDgfZt-pZx;#q2N8F&`{<rV&!AMd6e*K$=~hHmXSF`40fR&4SpQ0_+G3&LZG4uFj_`E!x%HyXiF?<-^@n6>RB>I$h)|cT{bdTq<_;)5Ds2g^sYEn-&`z^etMRsoQ;MF${Ncglq-s7}}%<SC3?i;zmGk7PHl&HNOx?Y)qj{(6l_`=;Up1W>qKVPy5<;0e;SVNUO@q``>OGz1tR>=S6x3xzE97#YMWo5Hcj+17O>eE!2{LWzasICaa1=#PBw<h56!^Lw}NVQ*&U$00bE2f_m{(P4qm@zHJyaSXpo>?Zo*yT}YPMcYW))whh8WO(^{tx*Z#qFOWNOezSz%)cn&zUYd9+UPwbf^kd1ZSCepQ<rhg+a~GUCd}m&Fe`*MChz3P6>bVKkLLpE%_kzSQZ+u=*bPf}l<(veO-eWWJw~e)qQwFn4EDtM-kKyy+Khs_Q6JF_*`3FBy;*?7>=QnSU+=oi}LaBqiEHT%o6QcuPd}e5wrM+4?&1=SLTi^g)fMq>ojRj2x&9U;mY6oa0*Ry)enxS{S@XRJ{50RJ7JEXp^Pi@tkAz$svH*1gdpZ9BlUILpU-@%2=!3{o_w&JO#uG%u92FXuVv4y;F8wMtB0J~-_TT5G;552UA$9q+e1JKpyMpmBMPEN<03u^uoFr4=m{u>VZmPcbXFcYYq8!Ln3qKXZ>;M;{;PNo6%5Zd>-^wMl^tsF^fZOu*MTmUMWv94!Kl56jgb)RVMS81;r_3SeixW0We2<uNV;IB<fZ2r2i^dFs;;HC*1@ajfwHql*|QJ?ztWji?*nG>4fHmqNikR7J2gK<{yU5(8#R^nulWSa!_J1Rf4vpY0=CUg7_Xs5Msn<(NoY=ShVloK~Qh)L{*BiKmV0Qf{3YPe~Niak*Y?-Ukqv7$htBzL2C$T0eT4Lr2E3geht<vr0h`1s!YKu8be^so0#4G=!7KfM|_Pxy-e_+Rw)<9Gd|wH6|YQxglk+J=%sP4mO6B`FRjLy~06{cc@jy+iv_tC@#<QM1rrego!>%#Dm)H?dWLvivrN=6$}Oky$8N$rTQ5v(UH5HA`ikv?OnggsLnvAF*i&)~}hET2XdN+x#ibrbR+_^B7nk&EQ!_%@!iag2BD*!77pgUtu%({J~M7UI%{<O@tTwp%=dVuY)ig{xeNq+q`Iai8wSYX@dh9F!lzGEs!w7ZlU^`|Jd{;myBQ8fCr8;nby_pcFxW^VgY;cbHlSUUM6y<|K&IC&M^&;BRM<USATDy5$*x<Rvai;oJ6=m{1XFVoQeRhb&>;NdEXLgG18Tce;^s`uTbNNtf&r$vtBGjfB^5K;feTV|2Ga+k1S}+m?y_+4^u4<2fl_eYT3{9CA0v+q6cR`@X|kMsDX}fbzaXmL0d=;c*c!^EuzS0mu;QsN8ichhKV8MS)@#`6spd3p~kRdxCZ8QQ1~TU5$PP=C}l!@RF)7X4e`E>u*B#g*GnCi2R{sTfh(tGjDnmygz(W5P}MGzhdBigtV6og13{D<2RE(}(H~bE()%uCIF&w+C}lxNas^(Y8(jg_e|=Ip0xW)?4!)@~dZ<4A!_q&>$>{+boDD5y;<zskTz%=)XTrdNZaxzRy0Y_WY!_XwTwYky5#^$G;K~)7rs_a#HudV%yQV>VEsu)?VGv&<T3f#PhjcdB#XJ>3ielhsIHXlTQEeID`k|=wN%c<*J?<bl`o#%U;3h`?63#4!<oc=>fvAks;a#K`nYz8wn1Y^F`D(C8=9rVX$l+$ZzDzRl2dN(a4R#*?7`n@LCH0b5+J)8mToE}GvMsP=G*73>AE|GcqK+L~rSQ3v+8D+gSz+uA<#!*@m1Ex00Eiwb9Fr7nOo(It9es}iIfwf&QkJA;_g8Ef#)h*p&V-jxN6{h~cJSGdBqTuYoD#{;*3&A|v@MVanGZXurY1C$uB?PI0}$O^hc$9q&%Gl8B&fGyla?taJ5hIyAL-3p6lAZUNw^oaf|Q%qRf6mW>4%1&w+7)}LsRvqPZr#l1w*tAkQ+t6VHb&#0u=o)K`>3-%L9jC9|SiZ984-ipP6JI1kZ*Ty+5!E$H65qR5jK24bRTfS{G}rZ(#0+71y%J<$`u$eb5t~D%Z_7(iwEmSrRD!<a&wE%Sj;n8p^QzE`H&qJw@6Wa3}@VXe;`w&E6<QI2FBvZd@85ry%br`&FTf?W2C}8BffgUK}DdWo0Q65C+v3UtTn-dXv{*6f1kDPjv8;ZouMCT@H_XmRVjc$vreVZFL1y?jeqs9Y6`|{&i65I17f#6Aww)xIgayeA;3zY$Iaxi%%sn(&f3I8(GCO$3^th@)c@0q&@xFSq1DVk`R^=qiX0q7Bm@3j;v=(x;)sg(=nWc!tVzhq*cX{sP^{{P|;nBdf}R`1IX{2j5AxU5}PeWu`PyQ0C0!+pZj8b+D*vYta{PQtRbxc8|;RiqjF6lEn7v>ASr82BAa9Pss5(HZqYp^-Ilaoz!b%z(RD*BtizBrudkxCb%V+7g}v!JB#_75NMv1$%09MaoNj!tw5%v^!e%1k93i?ayU)-FA2qKUPm&YJDs4Mfhw=|XnzJ(irkzJ@yTF1U1S{(Vs>3%z0k&+t-CJW)^?k9d$`$Y7rTQ~MFG<NX;h#_Mc*?P~!5;B<V~+@My^D*?8AgB|zq=>o3loex5sfcEIGTya7diVBLf0ZR?s@$EGYP_%APRaQ@SnSr28~IIPq_KrrFTaBqiV`3QN5l?-2H;2#$j)~EM<2jK92Q6{Q`%Z)p<~etV>+kR#X#W?pm*O${?5&D-2*k5jNaTk@ny<uL5XWH@u@plbw-+&8a}YC1vr*9&YsXNe`~<^}tFl^Z+F%QhFZUq_G9jc7z46_p!)!CxkNJU-Xf((8zLQcG#>->M%lZupXJ0n#sZ4Jy9bY;I?;e{fb*{eZr+Nn7>;eajQ^;&BH)?J3Qlmv3K663U?oP`&)0CH9z*=;kdf*z?S83jYa5s1-{w-{A*8n_Xl7^kxzaCwvp<;=_7Dd{Y##KnI!(}AA-B0Q~ltFy3w!37%HIJ9vF(FLGAfZudVOH>M?Z3N7SbGQdZ9VGoDL7@ue|x-VnVD6|*q<H)zY-6uQt%5tpcHATLW9W-2_aX`{6^1Ot_Gt(IVsJQN;*SNJZVxTWxa1C==*32|R+nu<9u2rE3;Ko3Alx&CjA{aGGiE+`pF1ScllLFK$Y!_XE-VyAF#F^IjJu`L3XzPq~smk|``x_6Xfeib(vJ>a%q!HX1tl@4}sKtA>ZgmK2Eq!_TWO+6+yJu%zP$FEQ!mr}NJfl)y>F@?z{M++3U269z^y-_;|oDC(2rgvxs-R4}lmq)c%WA#2v0SY*GJE8*lY}7a4?93=GHabspzL+ItrQ|+*n+!PA)FMSK*WfzMX8E-d68mOA6wu6iyZcddelilA<nzpHwYJ`4CXo#-6*I}z)pMuyx{8Wn7$&}Uiq#LJ;Y0oEjW)S<3P0coDJ3H3nnscqQ=}#<%e0gAtGr5ACM1I~{llpi$>O%v`gNPAYfQZ{c3F#)aH!&=$t`u%u-MLgvv1p#JUyzlyUQ*VD+PSLs&y4UQMaG%80*mx#d9^8e>Q%zEy5=C^qdXCt1INzFIy|jluL`P2rT*@Zwk%QuThCkaknm`yB+z{oq75VvOM==e6+fFb8u~B5uP{!XRBHYE2A*thk8~h$pF|D(}Tcy<VW=*S_tqtER)24Qsh$!ExszH2{~IHbL!DY_&rmkQD+GSScCIhFU_Qt4%4=H#$;oq40DdDA~>x2up#91Wkf={i*<EB2C1EE)ouGgJRQO|)ssD*%6RNXG9>x$;!ylfx{0RrA~u>dG7sLZ=HpVgl1Mw(S`P0fh*Yz#px-F=qGqk>1cX}#>$s(Gv>)mmR|ca5So7~&H+R$Vvx$GbTVnJ)BGce{!eU?Jpxc&1&fSJ>bSU!QE#af<L=J}g&f%I*n;Oit^2%!66nx#CvB$|;RZ!qrHOk$c5Zt-fK)?!3@D#Pj4W_y1=+!&KS$24E`^0I!d3x0l(p>sIQ<~XL*LUWm<#XqqIcdwO=JUDgbeXbi=6`3tyUJHrichwZ-pAyD6lvDW4gvVkRHN)wE=TTs|JQN#n2q|K>jP$-wFvI1>!#IYL25JpFHhbWF}g&36^4_r5X4wQ0lPm>If$p7hD^LYdi>(~@zGN&$)xPe(WDwgk<p4yA}l&p`c&<|6Of(b0LKVRc@;{K9N5C_33(myljA@oi}tg1!_7e+-<V(V_BsYCOrfb70t(60WjaAA)_95YQT7JX=iPdOSz;15OcF=0Fb->QcE*o3XJ?qNI#*&rb$@K7Gsobfl*-wtF{BX-@f25Cex2>vVR*><oEL*koe&&z8*=)zCT2K~y!5ZI*Y<%JTp2K!v%92Z>eu+b7SbIe&3dui3~>ph|56Zp?<7g&ah>0?{~e;)*_oU5EpUCf19rSeQ&vhe7b8i#vLqOJ*+?QveA)SeGT&GTdwy%V@}aO3pQj6qC4B>jJ71%)WJC%KE|(=`sUzK4PJ}vn9S8}SX>mqjr6<xF8jazwSzQ@e3W`vzf^WF{^a$9O>%r8(ld=mY#i4i_2A!9AUR|QSHAfeqBv8h*S_BAWGD1-sGu#@XS`zTHvo_<>vP_DqLFwHWy^agPO7-D|vp}ui|I;!e-W4dHaCH<@I7;{GDr^YlVZ{p|dPb6sqIAaT`TrIP6!AKeWN5;_RI~^B!2oN|DwfdD_Tuv2TBb#pSME<C6h5~z(FQ@0;~%PLzPM7K24hNZjdUF72xb!Kl9DfBPDDqLeXzs9%R?T~o2B<vMK`j`D1U*G>9mi0qp_u9{_YrIuuRR;G~EPACS(7nl_a>7w9wBLE@_z7aZ$2!D^kC76Me*qu_j~WTuf9_oz~8ii2+a05Dm^pErR|Xlm`D|>H3R^Fdfo&KuZ#W46gpFN#b<lya!3_(rPXRR98~cwj}<#FU9wgdNpK0QsDW5x2YTUu{_<u*yEmfQ?nMc`9u+5>_@=8BE@4J9%{dBxDAQ#ylsG~cu<U<X>MaqmqEBF<7&>uh}1%Kqt;XTZkZ%yOLC$I55~Tp>Me2+<c||Cu8|*$XB*7uDt%t^<r1@M%p8tJ1R#<Dk6Hp*C~4NU<FJ!*r9-g-Jn-D6YzhXc!T6Zzg_g|dOxz}QZNQh&!R)4g`4{V6+7On!<%D*!w@AN3?o}`-k6mj3Xyq}n(Uo2EiLV9lD3T33mc(r4-X|-MTJ=O*bVS~|;e<Q&K|VIyoJA`((*!kCyOV}9>t37pC-Su!j_EFYOyf}z=o*q#2Muwj0&5bXHnnmMt&Aryxsn7{KLd+c>vWIdaFt&N2VN2bADR?H9F|Na5K}Ra2GpgDzok?bPVlWR!19lM`3I$_Xv_)VBm?-a4w1Undl)b{EAXj&xL2%X*xFNFOU(E{v|X*N;#DUe5x3SCSY0_M&lg>z0$vh&s4zaOBWrl|hMJ?($cLbu>>ds4LmXW@x+?0A2}B4UhOPBARpAgsNYhD6Qg7!pS^IJa{rePg4ku-N;Kns)rsfW(u#z@1Xd{hjXx6xSJlhZWUaPLz8br9)uEMseeF=rK#}d5Z`xNcqj1b?bMJYKsq*4&oxBz;*7>8^G3~0Wj7w|5Y?Vfn1)lC>Rk#3z-#sSCtB1TUq1Ag}O$a+#8*8tR0L9y!|oWi82fDr=<L;m7uHmAZl0Vq2L)UcTbYiZLxz%pkX(3#07w36d8&24y&a)ebYKc2$bwz$zM9-(CzStk?C2ZV4)+X~VpXjuWB5~qU93Xa<7;b74z#8QLe(9%I{`rh(bo%vzzhNzt&6-C`zu4Q>v8zA-g&!-Xkt0*I{Lu-ij2yg%FYZeuTzByad0j0q<&;kzTZbvR@cyhtE>e;0}GZTQ$Ow<csEFp~RlEbb%K7qEi6Faf6Pl;zWl;x{Znt;51_2h^nVg)jF?<G0Y8v5ip;xat@DC;9W7_j19B^f!7(Vhf;uSipy%W&OcV<2MbBCMHV6-j^sVAoWjM9MTvKD>jJVz!2?(lATtE-%L8-6R%Acl0lNFm`al4B_q=Ou|FdB{A5D-q$CGUzRb4C!9hfEtqjN^kuY%s%uYbu(%94Y@L|MdJ>qlKF!)8pUb+PCy^fHW_pK)casb=hjvDL)LKLv#fKCenZ{>K7=YGEea+z*8XzHo?#cH*sZee={nW^bQ`zxl_y1nw$gV7u|0kaqI^PvMzkzza$xeQ?jSao$hlUr6ei!gC?9xQj8Ajs!d5;CV>7l39O~i?0c~+yo?%(XspkWo7w4@4<0a{aND)%L1I0F7oYSUV>au~%ty|G@9Y<Dk%8ji1)8N{LP=hRomsAW~Lfjw8bMi^#Mu#TSI;i=I_Pt2fwpYoedT?|4Tj1NygRc2yL@jZZ`v~PFan%=$Bjz%Z&s5s<j8wukO%wEbS$y0l!+f7nBLnLppw`dQEtAdu^7n>NZimO{XA!uTu;@O{_$-mCdM1XlWwFuxCS3fgW2JCdcAmaq5fn{XiKJ;G+UP?QNy={K+u2PwWuTIH8;jzifOkqN4^wKr}IYWkqZ`KV-om@RW25w*-C&)Rgc!pA8As?4g3fjaeOV-7dUF)rg{Wzz*u)4&DH!HW$B2AQ`!_n*Gfy~g2@@Skc$Y#Rg6VSi{tv|g)R+Ou%uNh5Ptyx-LDZ2}mu8vJ(P#m>QfRgE@WxYXRS6{t8!OYx%NhB64Q-qeM_|6An<J^R7rdx^2%meiW@OM@$1{ps_1u1ufF{r&(7>L4@9a%Mts5JtUV3lIQ?yvUlBP%N*J-(SBX_qq`;Jq;l!_$UPu8dq0-SIo+yeIw)dk8)KFcG6WCS59vX(aPh{X^HTcdgcrkP7YwOu#MY5n2<Gi3bP1l1q)8|NXP+_0OhPwKUj8Eyg^2aFiN-24(0V>ki6R8rYWl<BO5L8V9FRCS1hJWo%o&eYcCR$xD+$X<(%u;+W8Ic8h#<tDo`EjEu>f7s=VRBGXGiSvqRzFb<hyXo5!Mvv1MgE6e3Vj)@7Dr4`HPO>`CsCecfoDT<sw_FO$WI{|Ad&+8_>{#sBTCM5dkY{_~weCEVt%a;n=5}i*9i-^Ih1Wx9tYjmHfheVgQdBhXWZHOp9$98yhU_NfrMH&~d?zqJ1itYKz^dgHFJLE3OR1O?k?yejrE9s=#nVD&<A(E2w&_Anved>%Ou3{8)N{YUP_Dzxei3v?8niHX3BvG$!M{On}-Pf*|mK~mIoet)!V9TF+*jPN^rd~q@z_<03v!G)#nx0!DY)U%OK!c*WqC&`+Pm?BnrCNO*WoU{Hld`$YKMsPcERZi2mckE&*U-c5(hvUQ2IQ@w{FT4ia;HN-WG>%Vb~+f=%eu81bH+3U+BKt9f2lXEvOZ5ov8mz}*-?kpLgJH!gd&%3i%d?any<!DJe{hXPHpsi2fFP(*X%&VyGip&z5)&nhPE!^>8+nNZY*8a9Gl<Ev>Kq20bjsxZr!|^^3JT21Dmzk@jQ>Woo6K5PFM1<j_9BQG)ptS*Ku-WW@x<AzOe!(UXm6u;9iI=3hTAaVe+vDn(uVH%(ZV6;&l0UCIh)6qI?Nd`Kid)8Rg)@;nAV4UqOd8b6>dZh(}(=D{{$lIep;m>B9FDgOT~GGO&{_TBLA3JCiGdmnfr*#WPoIv`X6ORGOfbMm`z*prYiQ7wPPRok$g%K`yLuHU;Fj1t#}N0+|S8KDFhzG7Jxgt#@VAn*VUelQ3#I3GByaz35Q|=X{tT4?8@!Ac|XtlSQHM9~Jz13cSie*l4iO#_78!4nX^tm7=-W>m4tsy}dm;K9}jd(%O#TqLzf61hzxLsh5xQdG*_WJYh*MUzb+I<(6Zhbyo<cQB`(fCyq5LGkLLH_z#U2&I@BrN%D@m=!*-|799@8r$)1~w^zGwa0w6-r_mp4z-QRl2o{TXoD4AWJq^Dr`<|WB2C{Xs(WwGn0(V*{NWB%>Mnu_qS@!zk-FP%Q4KWECPYIQ#Zqg?VO{ziGyrBr_ur~a_R!C86ON~inh;~Nv7o)!4ZQpci@s?Dk!?fTcGoVBH?|swi)NQ2oWo;v-TJ&0d$Sw3xPFr@H;i?VhmfkE*C8DcW2f8H)8~Z};2j$v?OQX1;G{2l2G-7mP0UOv}eEhYeCXrZAE#LLjgDHTMzI;Y$X=<4=SxIY5WtAb|NzhZJy->zwNApgpuZbB2Ir!3Cr5FVMgCDwJ_|vsbW-$^R?(ZyLT5sj|h=Tw8`o&ZEU<elt*;~w|1_?N6G-!*9EB(NUS|U>HNDUHGO-wd)DK%yHWFnkWhUqXV(m8zLK3d-Nl+7ROfyk+(6mS@)2?wi%5#t^YRA}9McJTU3xNX+cG>a+BU7$1Oa3*w$jDSOHiLNLWOIBWM{TiI&D-fSi#)2N!!YzjbIs$5mejCHlVbfKG5mjswc2PCxrE<;w7y5a?r*I+0m4n`gs&8|A)|Jy<zG<KUD8~RIHkzrayH8o$*1Bi}$yv5%Ibn~|(c*SQi}Wf9s8GsNbXDnjXF0C79pbdyqE|B`c~^@&Wo?YgHx6Cbj8{%hIRdlZWvFCDX<y7}L*6-~XkKqvCtL>9RXM69-POYfW^@<TE?LR3e><oaqJyMe@pZ&VuP#IQ?|MN`CuxaDjz-UlGCh6zCX|jhSg>bjx+>QoR&B^);7}!)(*6WLov7dyfPw{aZ2`wA7SCQhetE<(Eu?gP&(B>&zGO9@{9IWnP=HrT0LES&WFwA8m`|3KWvcURaXpp?Th15SUIW9Z7p>g^2AEY?rMK%0EsBqd!WXTH4<8Ju{Xp6%LT56c8JQp3!9F+;i0IW}1jU+inXlp^iMS1&(qy>0eIzjWF7RB~yd<}=s@!)gRiO(sY{(*sc3j?+d4q!saw<U;$AoBvv>2cghJ?>b$hP_nUEhEVtxO+c^Or+RF`So}br{nO0J~hJ7c8z~4qPD-be2ysAVYQ$`YA#rxF8Nii6X=r1G&&uL^&d|e1L2go>Zdp@}1q#e_+1zYyDU+q49$OrXiI0Xj>lI7n<>kDc9$e*>0KM2&wG0fX&kzyaGsjrN&aqTp0M1SOb*OO5*tk(xrexrK0k}0yRq#j8hX=#$(iu@1s1nn0(hCy443|i|L(<(z$X%AKd3%LCf@};SRFY22Iyd>0Lx<;zp8?$Jm+)=o<1Fx<_vr_PI7NY7x+2jKss~DyOJ-#85xD0YKW+)?iE_L1fsZ%@g!|7~?r!uFQ?VOpD9*A2wLF5|XezTP*}gWY}cO53*tR2)TM_;TtxY6<cFbY^IcRB4Rds)AcOw8Iwwz6K&N|xC)C5Qy0#>w)W#si?#DTV&p`TFBbR)4}Ze>OwZ}S0$ZkzGBS>2sfvk`s$xB@$`){>5tG}c_)0uc2qpzqb})^L;uiTh`Oq(b&gIv`7H~h_(ouzbk&N%~(R!0>_Vl_y(E>WT#^*noxUnfz3IjvuVd`=k!@ee%*>tv#>@Mq=F{JT&dPSLZ5Mc9|V3|fXy>`6d3{H6#2=saTRx0LVqW);jirW}!gp7-8cL9#rHyL9>6xJ?@)&pX*Kv32<9rB-!rg5V!M6tEHF3v;1r?j-Qz~}rl9RM}OWEHo98iX2yC~xOb-*XYQkIHJ+MoZmVK1uapc$C!|IEb@`WM<3{(<~`Uq7CGWCDF3))^6hUUh2Ha$zUfPhGIrPixAz7$go9E^9gA>8=rvsQl;GyR!0R~l~t^0l%@_U28<ohv*BXQx&nk~R!-McAnl42V6;~AJ=I#4Qujb(kl6v@u2!2E*0wb|XUq6N?=#>j>BbxozD|()jLzkw-8qQM#ckxg-9@<nQvL4&1GF}Y<cZNFgI3J2(su`Dj~9O9Lk*7A1BWBVcK~+|p&~o11@*@MdwgBn0F`d?@Zp&F8~eA{9C+HIr9I}hG_CQM*EUP0S8aV$E=#L#`rU*SO(THf9f!am>T94O?6e3BrC~Y-%$E7kLE4bGN{aw|+4%63BIQa;@WI%Yj%8ieyK+sVgLMbTHLW9|`szzKb%bbd<1dO|*094{f$633%RFF3a-oO{XZ1XFwEJi{Ngs7uPw%@ETG$YCYGbAIJS8{k#&=^nqE>6dJyvBp#ju18rMMVoZ%m>r_`h$w!SQgkzvCwjQ6Eo}>dyU!8{P7W_Km-7A2GF%v5+N2vd1bT9V^0%&+TZ<RGVirc~Wizh%+?n_X*I}z;;%Y$_gFrdP}`z$~K}MHFwP6jpoem9@)89olUZ|2&R?G`#ICy;U;N{%jb_k^P6TmrpydvV*cD~53T2{a~+_+eI{v>RhK+P>z+ke3U!BZcH8J7>TR9GGH0%Q`TZHYG#8$FnnT!5nUysDi1bDSt-#YUl7i(jJhXLIa4*a3Z=y=GTa3pG*S;6*>mE?T0p_fvd+EiPggBXi!GsC^$2Wh%hO@Ye`=}m4{oddQ)b8OBW*~51S`gJ6ah^k;`86=?fierFUg+@s1EcU>#*2;bx&KuuB;dVzL<$}GmvnYB=G{ViX-I@rWwNer8DHVx<uX~IX6c-i0#%BPDosZzO<DW&?D?z5FJubVuM7ym32_#;eGAxn#dwGtK+4XUTDHMJOS7mC&98L;G@SlET?zY&_75AVNEEL=@z~QIHmX(QRkvo9UZ*%AgQIPf%1X5qbS0vcGailX=sB`EX~WwaUsUZ)uQ_-hrsp#x2+HBemFPZRKgTUZiB~TmkYCfkG^qiPFC50_CBC<-6NmM@qsA<MwPh2imNo?Fu8W2U%n5hzs#!WGtQhN8d{5YRdJmA~vm!mm^-xbmPl<JcWo#k*7fEiqODG!Fwf~abw&>YOPmYrr@N85tBdKg}vQK4lhExYNxMcGJC%e}S8WlAa3$JgFPJVs+Dti3u)90_Flh?l-y*d_-hMwlDI#W)zt6M?w5yRe=psf~Wn60W?7&{l8y=Bd|CzeR-82qtHAKN>{dt1h5<D?m-XAnh0Q+X4%jVr9>a$#h4*FYQQ-F4dTnZm>C5(ioZeoNQUBrLJqsVvEdOpM{wrn08W@4bren=66n>{rCHrpkt6RaK#+sUipxUd2taXsQS@g;x<unko{z>ej@frm7|ip|`Bns;RocAh)_MZ)#*e>Zu0xU_1ABytKY{>3oBsYU{J9AII%DUnXMn7XefOZtW^5nBQ^K+d8g)ybzomY5(Z;ap)#7myXw#3Ddr79wy=ia2l92i*3Ub%@15A3sg_zoNAdMtIHDUM~Uv{Fn;Ct1`N;B+bmuJrCev2lYO8kfYEWLm>u0uAH9U~MO>C-&aRxnD@<#|QRO-6GU^dmm$tXkd4gf?lEz@~$R^S=FzU*um*nnpo>bQfMX|sI#K$jmgsLjaE}!{uP@x2LSQ-ls@IDlnIVdjp*_tv5^Id`Ln%OA%^1SCq(qI)maB2xTc9E<5g5T@tff$Huenkl-G$?7PB4&9lraj)Q@uHI6Rrxuoq&JDQe3MLKNv>GaOOQfWIeOvp;|K6M?&H901|*ZLAu+QsY<CL_pk9{DT+uy^D6GNLph50H<+Yxa3NuQY;pTT8XF%4E`wkdwb^mUJDnOrWlhm62xo<(Sjk54#-*Or6VS)D7rln9ekSgwu;j()seUPrM%mh1H5utY*QBhZ$_ip|7opq7yd-btsOB>dP>UO%Ms~KDyZbwkiRUf4+w*i)PwLlGu+X73rE^k!t$&_zzU7MFK-rlx8VpHDL8g5p%85C?;`>57N7H(V7dP7y()e^3Bw-FR{)z^A3w-pw5HB)X?k$1d;ZJbuP6IA#31>OAEZXn-#;A$M(j$oXF8+7w^yMeS3IcVqob^~T3P0-FC?gq?8rl6gN+zr?TN9nMkwJ@5U@PM=Qh<UZ@s;?O-x0Np5(w9ST<?r1NhBeM$8?U_E3f89*?8a4hf+t4IT^tTsv#pJayCS2%>7pGPl5*2;N`*TuggfM*TT0kIc@P5{<`1R255@)&e3Bp%Zllr~e*J|+w`8?+8<}_R+Zq&pjT-F?%_dtdz{WcRb1vi-Xa~DbbUtqbM&IrNsB_gA+Fe%;p?0jE4Rm}bpk1qIL)~d5IbCafUb+y~bEF19%FuQ&N=sdbWW8PP3XFLG8<NMBZ*30;)@`25c5bb#C{<(mTUPnl<^@EOFh+rLeM86|xKP%S%1I<y<x?3>1qH-tmKKNNO^lY5MVwv9FV}gofcqeQPv&Zsl;%F)5Zuzc!>FY`yUHty;b2K%(zx4je<M3*j{dCJxaP0#+rLOAtM#jvgPLpb;c~ylt)q=w8!$7~+=t-}-0yVhVh}TVJ2rf68He<=vCBF7SQ&4UsK|5q^olrvIedGtMG@gk^nuZ4^v&#J&osgB5-7^pNft294W&|?B5yGV;aC5DB1ViHjo-BavER-yyw&3#F0WwB0cy{)s-!TMea-+B<yTfvVMMxUkuJ&8w`;su?_5UiHbX@RME`TY2~KF%NcDzo{n>Q1#i&?VGbODb2hcL3?_02?3H^DDYLe=Y*)p(H)kFa-f7x<6p=}ij@94*#ohu8D;l%0KYMPd);hLbzDMghyOb6*jXh&*-F>!iR86-qI?4`3C>8pmyq9nr{vPiIS)NYDm3`Fz8#GSsE&7(+SqnSOIL7|s9tOxxFq)(jhHq`C)wzKEk&faf#a%}iKo5<SB+?JdxpX;yDm1f2Fp)rP(?dHaf#*|wuC(RYG&4Titc)W&dGr52Xl<2vR=LmJkQ?$IM_)TcFudY$b{D}@)Icf&83$aE$G^KSIT4}>f#6HVqLDLWTPN!8?a!H8Qpx;#8#@UFDwKPZ5y`I_wslA7{X3lnzykCoA$iOGMn|>{HUWYMwJ&qrv4~GczKJ^<g2jihS7>Dsun~Y|3tW>+U6+yM($wgtyV(b_d+DPOcCx^pu+B6vAHaGnj8_znt&A-Tgw#G4iz8$T_VzqA7Zq}r6ZM$Q$({Qewntpe2b5GE$wX>ROrh=iZO05DkEmZ(5t5f;&aA3psI702opx$Qu&~I-3O2sPPu^BBLE#Vh2{-7GW7<}B#*rUEQpYq;cv!T;Wo)z%qusPucBNY9qf)6)P;;ziAyHMx)>a={{O2v1i=3juS+j;dSMvq9*>bue}Z@mUS^*U&_M`=GO(%jVS?K>-w^l9tysUIUlRxFh9XprQ6zqe!*FDe?c=kI*U_|&AwFn3&I3KFk9i_Q(rS>dq*g!fIY-|y6t#P6oo^hCKE1Sz67Ly@nRJ3vCoya5Kj@^!S1n>HZcw)SzYyLgwLP)p#p!8f`c=B88#4f9hk4INNiMS}&y`LGp2eVydN=xixf71U~Mz3J;?Vgz8%t}vVDu3UyC=-K+=qnFAexkVSe9*VIaNcS~5n0-_+{69c9hoc8d-_dNlgP=JRbI2E*6f6NW-Ea84es65Z1$myov#Xyr*-Kbmfshm|QVecF!HL^a{~+wmyHOol%WOYvya|@^94TP0H~c*ZSp_Bbxqzhv`G1j-Jt|p~;Ou(CXwYM^rWf@%GdX>LzPv!yE7I@;M$v@or4QKG$B4~MSGPSC#^>P!6nY<is~M=#X*is&*1#;n2v)Z~&U8Yz`Nrm69M1A{x&pbbu>)XGTfi#KCmRN#oDYmHmKm+B(Zou{P*N4?)Xi}~LEiA(T%F3K1pYDNv;_5NoPn39z`b1h&0qQIX!-cQ^#c+8_K%XDw^Ts7>3xdZ#u>L&M?%^HUy;gwL*fWc1Dih9h7ekcRk>L<J)6`>oTnHkYqG$J8YRKVa8RwPE~W^qZ29m}E8njAKou0TO+?o*&%*qx#O|b}a<JKW#08n>D;)a9gg3*mco`>AO|uIxaT{fzbcq)c$u_VZS|zWNToEXhtFej;%-jN;pPJN)1eXi?FnnBGtTFu58~jzsWnB&9*$mszCvygR1Ij)>9ZauQRnOL&&g1o>n)LQB^JVhwKBt@C|LyTn6aznwZ%nYISNq1uuG~k>WrS`7qtQ?UYwR9+CGK*7wu<Pv+}xYg5AVkX_C-eGKw+rm@b?l>unjny0S;zq(GGl-6aXTozn6J=nack@etUuwBb+34>Ao~7+~;_D(dI9a47||Ve~|!POYFXTjr=@XnofFLdmmGz4ytvQu%Tsa8!!%WdqltiYxq4%MSR5o5B(M!d_J>yk}B3xqmd>iz9(}+j5B|g2pu$D#)O-r*7FgDi^^9OEcyF{!p~JnS0@;aB0=x}WEKvEj5&zro~Uvca08rkx@8^u$V_h7x{M0T26*jkM-x@O6&@ncj<87TgjH#U1*c$eKHU)-9+~+9<i-8-={AU8dXR7q?mEa0%$5+p<RI5sx+r%9Ms@d(K@Nv|q<oH<pE#@@omm?F!*93r^a*3<ep}&kl84fUr*N^4t9m}Ae>#?xBm2P6N^&0@Ov}w($s&V@6o{l8B9OKY>0IK7DP{ycfUDjf0{w1lMi=S;S+nM*<Ab)AKmrE9*j)kDZEX5oTf^xhUCH%nY&dQoEmFQ&%Sc-rEeh3uL^spq0F%qMG-6^rfQvM~ZD~Flj>c`x(XxRi+R|2Yvh}8zbCaUW^)Lcx4s1#}dDok-zB%Zjoz#5%E&e-S!M~>#M$v~hn3w5nmSDb+<z07q5#P3L@VBkKC2*<T5hvgVY#rV=2VKpkpD{abPu{h6AMf;2Cd#b#KwG)p896{~>25t5K5T3Kep}0f_Ld;{Y-`<m8bFfiZhO!^V~}?~dA>z*v90Z>b;@WD_R0H1jrpl)K(=vh{XclP3kVVg+xBcH2xRrv2IAXp5DQ9q0<-8|jt5(ogTQP>g`J?_Y(Do4#l>>H8t5>2P4|!r*oaxMR1TO82C398w&(`d=vXTP=F-u-jbeAp_=F7<Ry{y<jpadyw86YMMh}>;$&LQy^Ow(+lYB5f7(Ns~zYmE=$B9UDH<dr;QI`-fwsXl$#DSC3Y9hWH36MF0k+Q5Qt)&UuStC8+ZRfgU($a=m=6=9=xtV#$Tcd3R99N$*p){`c!RWzv9evZ9;0<t&j&9Sqn59{~z?5l$#)3rxgzJ&8WKu>9624$1p@;SrT2`xpN@qc?$gNA%HB1c9sWR~30+Ih1F*Judrw}Ek(&>Mwum45qew<`K?OPcS@M-Ilc2K5)76<?z7!h+ZP3|hmR9r!cqsfeS4ABe@)%lw2TV$BQE)lju@5EJt5Uex}<azl!mo`P*ZI=oF)Bv9uw_b!hBd*NwpfVC)xalg;?-X+~)l*iqj*+aZI;mR6*hmvLV0tA(n9Ik}Lyp%f<Go^TI$T_<GeK>;b^~Ti8OJ1%V(<YzEdnlH;0Y3nWU3*saap^_la~V-v9;xL@0hC{zFh}OP!^gdKOaRqh|%6}Hpu4#IcoGy?wTvcYd=&gxW=gbCdxa;sP4OQ*f~ObX-8l0f^HGqN8+wg*mYd{;|c^us074{-Hd@^WtB{xLnSh{4K^F38K=d@DuW*|Ye`ZpVbQStp7IQ4>oi!?44sN$Xla#Fa%}WKM)mFEw~r8e%+v?uFpVLAalKpbNIQJ=C+)U->KGi)_s1tkbl4tISZZ=1ccOBaDz10OhRA3>7?Qg&gm-|EN>vt@d_M1se1UG?m+>ae*F`&q^FRS@wlF*0lgBA?oa9X+7hJAK;6~KpKtzLY|D<CPZ5tX}RJJpg9ixgRHK9Dz7_^(W1&<-;-1Oq8e=iL>#f?D0S}fsih_fx@i&tL`YuvC#a(pIC)Qd3&i0X??8nf>|$4crz8;%99g0Ier1hQ%xV;f2AW86}Kpq?JRIr*953CM(=9pH7msRi0DHH3^ptORTfx33`u2WnZuzjiJm+#N-%&<*n^ItV9IqB?8n)<9dkE(Y~J4Gj(y5GU1UycKKPd6seKY*cV{hiOsam3gOrOo}XDFBU0Mb4{GL49HmeG~d|=o!n@aF4s#nc@-RjuQC~FtaJt=0tyXQcHFFt+k~La4;^Y~c{xN9#}Do5$1E)}7>xcMpO^S=5JmHJkwj4#_C=4Q^rJ4z3)J5WJ7()7Q*crT&_hH^f&h7;$#f0o4(txXuh@bC1qb10naqA}X}zz|Abdcvi3c41xJ`f9#UTF#Oi&3VPDX3SNXgI-8+#M)=q8V_<MdO!=1vwua`*3{p=`5;+Zt}ctwf>R!gL*UGW`vC3Q!0z$pjE~e4dgxMJ(fL5l`{Rbzi~u>~*3XJIg^JW2lG1%H|sEG+Px`zClpjNi260F+6HdeV|M&JGnnUKZ2JPfdhr>v<O`jgY{l@y&H^2qwOaVflvx0MhiZMBUX%1o;MJPZ~4`*6wltieko3VJ`!(#ef8CG*DRkse|vl)j^4a}^0NrkL#lpnOjh5GMl#bi_75XP;euS9Fmq1l>!}<ltXH!7>uAI%59b;obs_?KSPZslYPyz;+2Uzl6zf$bFZ}{jaZBR;m7QcEP}1M#J3!&Gx?Ln&n2F`8;>oBUVC5-NeAq40qBZMhXq|LiNu~Ot^bm-We4^o!uL-cQlkG2(#Q^O`yB7K2Vb?NG;}r=*zPpUf)-hYt!?ooq!~8koO_}_$#w)UY#c*WSbYOTaM^3c4#*cZvNaF1EiX&P*rq5>V<k;827NZ1$k@VQax+!CFZ33Nxl|FvW$lBGA0z++~53P6=gQ!7EB`t*R)H}8QtnSIOdmxn;-I&y>go|$m!~uNT3WAtBYd%!1FAmpmkj_61p*le#y+eJM{Q&*y`J3a2pPwUX24nI{_PW6Vb`Frgv*DpOZ}Ugplm?BOFdXO5wsG3Ca!hG2ooR$7F;;#>igx*1yjaV;QjTSlAo~=GzcuSQhGqy*S(%X)+MzQLe;BjbioM3s*?bx=BL3T$EGpA94B(Yiu#6Ejk&SVa*;1<<(%<(9l(dG!G)Yv9>71n7U$z=He_;DHxGXUYfcFiSipMadOb?TZ;0zImmftJoWQ0N+L=-O;78p|zIPXYi1CEw)wOlVE{p=h7hjD>QP-TfnuGACquV<esmQk@-56^L@b+S)jwOw7C+22<MfI&~+T9C@QbbyhSCLg@0F?`fNlAVV<_FntI4NXd~01SoG%2L6{{`3;&lg#=zaaCn~`G(VLbh7b#M@>>KoU7<%xixN_bg#k5kF$5SOr3PB5sxJzy&R|Dl&ipP<q7~#s|RR*fWeLBfLPrs;WZ5g4awwC=9H2#USG(*RFcGxrWaL2iZnfCDDByaKx|t@p4B_19Yn2}{+u%BK9Mu6OqyHq!Nre{;)7W+_@PGp(PNc*$SF_8k@P)^4%jeCs#_)xA3P;v{MqE;w@;f^f(PC_RzoT#AI#q$*;ZASvsaHU_$V1-wV3qdqdIe_c#Ed<3v!viN+M|<ghg}!CBYzj#u*Z|moTemIG?9bGf>SzvZgvjrq5yL$Y78gqkm3!Q>f9@u>GueEfJ6PIt9Xa4U@)*0c!Z+M;Z)JpS+mKZ#=qCF9_FVRsu=UQ}qs%%#OKGgX^PdVuVs-I*rSTO~!G&6Q=Iv<$69}BusjUyv|mPvfAgH0dn*K>=IVy1^qDStNV04`FYE|9XEAU-|nPjvO238e{idr&p1nIeZ^f_1*X;tnl6(i`jxqCC%#L<A_6&Xg@VV}G9NB-AiHsC^REV#5{*yClnYSqAY7@?Nx-Sw<Q;3a*YOJo-EH718h+C@SpD|xg&&*k<v0$`TESOuQ6!ELYe7*qfwa&M8Jfy=jpOLEYyvwGW65@)v2Qf>Gup~faQM+q8f_}`RgC=wR$WZ|ix=FKIC~IgR%<wi(l$6kC6s*}S&j9U4RxW*>1B#63aYb!0pdq3;(Qv+bk!RaXJD(W2sb)#ED6+UrDMEO@-2$DU#5$M<FFc2WMipF2KfLQMA*WD2sE6s1SZ3ap{QFvghO$nUg9oqQ6(g;nqk>*%?zHrcy59iz<nq=JH52*F@>d`!<6wD>9NUJ#3o}sZRFNd7F|8%H{R>3fJ$sIqVp>5wZvYJ-Rz4!i>ctQBB_1u*|%~ptVNcmk;>^c`0YC3)8Jsqn(wZp)3u6mv^8_Qp=V}Mm>8JfR)7>*D*n!Y_pR)|uz0d5{%O9b)top!`F%G*n7hIgJD#Qeq2ItN+D-gYJ2;Hqiiywg*)z1<cjb@1$Di5#*(H)L8V?hFX_|xIm`wnVP6=q=7wb%0EU5TYM3Kl|KSI4_ph2PU-3<LI`_4JEoZuaDh1aupNaz#i2nEq)>_+!)yyaWCxOPsCdz|s{{a_KgkK%9Anc#yvs;{W@tMFrkx(nEMBytRm<VCSn77)DSbsk@|GS6XS*O*J)tmx!}_2Z-XFSUoj!r-MRf59xwmda12ApF>rlvrIJPP9gPoz7;2I~@UXpsjj!Apxu8yTyJF@pjWfGP_noySfu;D~fDm2xXpU=K<$Kt;dh*-G<{cko8^I>x$pQN`q%`IlaoOXQ*0l$(1p0N=gkJr8^{C>^VX~ZE?f+nvCaq#X36_tJ~@_Z%tU;vg^S9SR_Fb@TX7T1oaJb?;n{wG6nMc&S)`C`@R<%iS6w1Zz(%mc?_8Jh(q2-CKkFK63YxZUzx1z<XSX*Jd2lq59*QBf$=LOMJ*IJ6Ok<%m)a$g)q>#>%O~?>E?2{!ig<QR#evF@sdHiSWg%4<q#+>-d8veQGi556=cED|f2$9i3xF)@nH>$dL3e%X&`1%Iue?5Q;tNTUM(4TZ5T(miUQ~kW?U`)K2#m^xXj~IW#oolgjARjTy|DQ92SCdR%P_WLHcu}mo48==fKgP^Vgh$aAQd*D9Q?&1KSx~>J~K_EQUkdtrrSCzvqcDHFev%n19yt&rRfVj&53oKYNV|qfC2qCbdQMRY-*l$>0L-aHxBg6z5TwZSxQ)&uS)qCLw#7l0^`O9)k^78hv131rgxmQkoTOBi-6POMH+|0Jzrsj5uu@dBv+}Hh#SM0=hs@XDAg)oQ{XK$P!PNjV&L5HxY(Fzsop~u#cb@$FlJUrK3gL0t2lh}2Ios~-&JSyJ0k9h8E4msvT433eb-xx*9hxroPe2~NmixCLxqbvBp;6Tuu(yW0bqFaDjuS`O&R15la_i7iRYZieRqrACrKs+FXGszh}l&R8;~X55~#cneal%MUyIi=61^8ERrbizfJ^Rb`Ge&iPk85>;|4WQwgwr5Xz`H#p_tI&P^I<`!nWFQgz%3jfR(6M!c~dTHbE2VEE}$?bO8+7B3(uLd(?QzaKLn|gSA`7fV<w@zZ&F<V$p^>B;}&i8XeRAC(j7+iKgS_j#5th8a%n}%Us>6%#Wts<VUY7%Q(2^cXNk}!s-b}NAuytp5?pIDc5it+t<wfi6w`aQ(r5#qXNjz>u~{U1clx3^-q#Fu4Jt7`<4&?jvOvf9J-xWd0EZVo8ZIMA%@b}91b7MKlZxDs3rV>Gqv<}5|%z^Bk5zVl<;HN6jBA)fYz(iunNG_Uj!fKtOVZwaP_fw2t(-@6ACdOWObGd4|%CR_GuR;);{^f+6w7$h23xn58Asv-b;43X8Bdet-xeZ5m^v8pWEY$u$nbrV`02WnlqhI_!D|$w(eoN6!=d2Q(Lu(eFIS_87W~;Y~IJd^zE^9@O-S#T2-GJpV|iHgjliC{mdw6v*~PT=@%Nz?^v4#`)XLmdL0L48opcD1GTJQZrzVwEG2rc#CSD0+P)Lp+2}IsA)YC~LeGg!=7?7^AaDh=yC_l}R>MLa6_Q->Myzmdz!w*}9PFvJLFCzrGG-{)-a1+z8?Z_Z)18{zuqC>>QLQ&dSl5fK9>|-{*%wq5NwtbpsMICv`=TdgNGD0vBA{xMMsf>RSC!0Mf}zMw7&e3M?hqGiTX)u~w{%uxx3<x8AmSj0-{LkQ21t&3Ur)bA)ElS#vFRGIM%#RcSbZy&=mRZ8<89I~`G)ID-b)tfuFGY|Q`@nQ&>y~ol$WyV6isQ?D>CAM{|7;L4_Zeg8b=YayAh#t6iJqstw{F%HxM)(zy",
    "softvq_continuous_online_train_v7_1": "c-pO5ZExE+68^4V!F4{I+}cW9U>l%&F3@YbK(M`TFHL)gD++;@Xq#6_)RL5wbg}>ao*^ZflAR`tJEKS)ayT5$%QG)I`QfxN`gA3$Q&H_$eP_3-iYAlE?mSt%U|JQ0YV2IARbwpMowFNhw=Cn0;RUO?)FPkrCYOq3s>-F6s*=?viK6SRG>raLR#ddii{g%PW<<?3w}N3+-?3VOAYts%f<Xx}guyL;G0bW%iPc<H#a$$BYh@ZO<}1k!D{;?;6H_nF?l}&Ct*wM^r4d!8a-kWo@=jX3MMR;LGj_9;g<!SNC9fb41k9!3s{)(MZUy`F>OJGltt=#mtXsizh<c3;!%OhmitMX<05eTpD{Ue6jlz@HLPG$t*NK5DMXxdDWSlWY^3mH0nMu06l*ZuhQv<QUwS5vZ7Kh7(%o0UwtxA@r>&8MKY06|tH-k~7tka2!qION0?weIQ+xF{5t?ll&6w+B)HL5Wwlnm9UPU-3HIfIgS&mT*E?#R!hDE%8Z;(}ZJv}_3u$=_uue=eXbT8&cOfJG@%t~Uk*?^GqCD9Q?M3`;+qFWR$l$Y}Z^V)*1@jaS1ZY?{VK6zdtw)-a)g^6U)Tt7fzOnHNU12j~N{2pvz8{@~#e_jB5xkZZrqmP<Bq@0&Ou8j23ThPIvj`^)HaqjZs{Mt&1`UdeCc#f%~#RwhAsNv*16gZEAT`Rmo&i}b_gzu!z|Y_gb4r_%u+V?}NDEZ`09%~N)I%Hn}6$%@-->#uwg+&^OY+Qq;JSn|_9UiXH)*1R%nrAwicw2QbDO8Pu;5~NUHDr%K&Cto0qTk8$>Aj=pI@E}oZEkc~<y~EcWQT>$QrUfA80SM?|W-jJwjiQslq!d<Ylep+0UW1oF?B9D$OlmE5X{DqQsj>Ow1v^X57C+AR1O{)$U2f;ucmuB%23yaD>&8NCuO_!UItBsj{qExIEZiPx8T)?skLW(He?nHre^0*N(6_+b{le!-gV*hN?M&t(qNg}vAB||?QmdV)7@^7BWSX3)S3lh+c_F`fjBToH%d3qb@Vl%6W<)Qurr>(!;IccPsiG+>!_=CUidc|N|HB}cBO~x+|Kd9-biizMDImqs;>~@6aa@05--K3#(3BZKq>SkH0(J!YgtW|TxvrV25iuQaOYSELVP)lRSxfD*Te4u>T2a<ExgSHu2G~<V)&y|4Xn_$94vHLBINMvZU2Ki%UY4!}2Oc@!^!oaql~33QB%Zt}@H%L+cF78W1Swra<#?2ld8=w{BEwx=yzee{2~38X;yWdCuWjU}CW8Tm*2qn*3RidFxKm%dq)5zeG_P?X(w)pg_gV>DBVZZ9Ry-rw^DfJ3VSQBk{9dW~M)SN6a3hd|;Y}tPU@@t*3ugz!E|VqFq;xtJ6|%L+<LMBFecXF{gkUfr0=BwX2j#j@+{PaFS^)TR1d#WEIz;#5fO!Dvgc2RF?~je}<I5B<KH@dt<KW~Nn-FIn<iay6cf#@MHFfB@^HA!+dDtdo@mF>>w19iD>GTVf9qft%=Yxm5{QmLYQcqAJRq-M@n+<M14R}pQ7a8#w&O=Mxqm8Fawur`B=y<)PM;=~P%3ECHoffE*#|ISj2|_;sUe)6B*_YXo)df};v!g6>u|p^>!!v&AkbDYrwPGLaow)M03rcAMcqI6Ch@Oig$h!gD9N`8oRE0NIW!zAWk9OeJ;V#Pk8k=dMj7iw#Iw0ZzICF#<?V|eUxUH$Ey04UjfmtghdW;U{KwANF+LTisVb)a<bge@ok#7Vfxo{nTa8eNHfMKr>$~Z}MT=l8FtEuYSLwaj~gf<Qib+7FptoIT?+juVyZkXQBDQ_k4d{Aem!>#YLah;Su+Nw34&f0~JPy8L%b8MHv_j~-*kqreqPT4+wIo8H;YUeZ^iEjjsc>ar{XAz{%0738^r6X5J$#*rCL-9cd=u1Rk|2<bZ1aG}*+1D0bQ4p2`PN>5|;9jokhRQ6l^Np9;%C!}F`<l4D@-r2{-Qwg+;`1KN?3#;vS$F#J&Tz9ucN=4E*55h6szW}(CSfjd8{+5*`$Yi9_gM4nN9Hpb6SqKm)Ok1>)~zV`^0`Y+u37kL#11ymifS3QQZyhq4;lytL@JH19{Jk3cB89X_rrk^!!7DjG@};IH3WdZnHZK4qecVyc$6_;A}hM|bdVfT;3*WyxiFciauh`-d@Ys6tZYz%pzPi_;oV3fQ@U;dU+mS0*}5r;IZY6%4K(3Ia+LupU0))0)#SXdCRYmZN)r!&=!hF%>85t03Yfv2-)Pm;Y{RW^!yZK?H@6MVcu?Jm4H{KQ)!BE%t`HlZ-TBIUCAR1ZRHFgzg|pK()B+3UBwGB|2;_nhx4yZc=>*rVDqvj=YYwXBWSq-aN)=>2-=n*lVsA$@>VCX%B(fh<4UL^;Z1zXjMMFkUmVdS;!WH}D_mnNehz);x8Wb~CTF9>=4r3oBTF-nR;}4_VcblL8Lq~AJUhTLn_(~Si-px%dGP#x+6Xi<eIanE2F{}i+Bru?a`@+$h%DIm`2m?eR!m5ouH-bsLk5lVI4eETyiw1ktEm37b{n@^18oC}wgN@X(WJe&n5;rOFGjTj>C@35C3@Ix))$9wi_=uzSH??1$1|n(Uip+Svn4<iIA;%tiDBV$!!PIl_52N*KbVa056j2jg@~<Mz{d|x5vN-5*eeVWDI8M~jBv)^)KVH2{-@m%Le)r}oef9C;@>d+Z){Pj`(4Z(Bg|zZeL+Is!5{w_GbxPY5YK3ZYFluuf?dTHng@>ZvJ8f(dpslRqsk0D)qC1orKwE|i)6Ey-$!y|sLQg$qKP=hrlcth|nf$Rg9b6rV{fWHk4YlR58_jp6dFz26EBW?=FU?=LLVZ4OuVbF!v==^Uvt3&7J1XCo?Ad;=9UTp{<u<UG-Uzwb+5o!#<WTW}QJfcUuzGYb=auPyy=OqzlemTdVOrNQ3RWLCy>oNO(YU+!q1ON0?_EFu!&}!oVOL$>Ldb%WV~*TP*>pc4opT@~uu9Q9)Rn$D+X!s5!?P-HSpkJ)&ONMZQ5QT5CTGHX%qi-gwcMP(etQ}IRmLFMBguQHddw&nPkfN3V>bB101gJz|BhrF|E^shE_I~)P^4_t|FY*q+J&kmO;an}nx<~RpQf~xrW3FEmhQi=(O{G",
    "softvq_continuous_online_train_v7_2": "c-rk7YjfMU@w<KnPMPUQK8d#6OERO2Gw0gQU9NffY$s{!(QqIVl6a;_g&^fP*XwWZ?gAhIl6tsvo#}`AgKYte-Nn8afS!DFSjg;f8YPD!S+mtczDSc!x7%Gm_m0PG#iLAw>_Za0xn?jlV)s$LU;!^Ak6|{;6Ol2Q^Jx@EUwIy-iP!1eEQA8C05ZXsELyIzbS)%{qt)SDq)U-!4~z#vk@4VRz@|mcqFl0B#+PCkrvZ;UGr{vBgEaw9SS<J*TSW84aF&VBg-C)2w&qzxgjtm2Df^Y*GoFMj5y2f>ro<6LZjL*5LaZdq7d&T$6tf~W+Y8bpj1V$P=InkE#eyyQ9e&6cnGh_`c!YC98HIuc3!cn{9Ctb+kKtC$F5gDWsNAZAVJQJQph%1Pf^jyBz92CryNkD{Y$|4H2Iv=}!&r$1c9n)A1FjKy2K%XS$~=l=<UVAu6d7NOOw#swJj~LVFohx)0gerrOc{tT!7U_`_9z(#nn{`iSj-<p=yi@g_GtlYYx+`<urN%Q@dJn?y92sK!q+^C`4r&!EC*=L?s>Kx7AwUvav;7$z$yq3G?pc>Y8gq15-c~loQg0+2-p?w=X+p@&G;Q4jpN~0k)<S0M5avuAwk7i8WsU?jkI8dbR$`spkhGhs}x{JI9j=A42HTb<|vto3}h|5&YwMYnt&bV(K^Z>4(Fh-)sWxg?hnM5KuCFjRw`nVDNBn}3HlWv^XRKU;vgO{yhAiPq~QQh#RJ$EY8mvEfhn)h++gQF-mvf?;matHu$zl7E0~k8;t<?})EF&AB1@Hlx}qr}6+o>3TQ6Y4;ehdV8imSO!EVzcUkv$j6-Rl2#+t@4UrAC5S{?{w`oQJ_BoGBmF4Ht$ke*;iLMq<{TFOMgiSA1cg51C>LEn=6>5v?}5;-~+tR#cZ#1fR!iBle;nE;D1gcWoGWD$Tx5)#H0=pic-)GS9<LkuNZu7V80s=$N(5O^4sI-OaTE}8Go3N#7d$C#4B3Z8&Ple}f8qsP+f-#pENh4aBn5^q){fpQ(jC3~fCm$SIxI2N%XaSkS(MG&Jzc+PWvX{KT6`T1o@N~W`X{n-cC1P3V6Li%7HQ6hY@RDb;p7p$M{3;vwIpYILskJN%ur_=HO!lgi7rO4R{4tn-b$DhI<z_nb#rhGDLfO_WHBOM<Ul!GPuJe%YEJ8(6?9f}$AA!Yb8P;=e`GZtfv8w0S@886~|%xI?1hA-8BW3)jC*Kr{LO)5QD9c2)n=R&RT6F@Z6Fklo&OD)5nOi(D(_&e5+;(?a+ZQj7tS~-Jp;tT%%7r8>}B~L*N>45nkpC4=K_iR=46+U7dxQ=q)?@1BQ1}vD(A&R3rBmdK4e)5XPQuHeZWC3o}>wD$G<`VkdFIT{3Xqugzur3)+m%^m0YS<}e33s>nbDw5$=t~HFAh9L?(i;tEuQlOyKfS+tbM9Zi{>Mdkz`CPuzu$N8$Q<)=i6RyW?YYkm4_VLQ#hdaxSSXSQ7X2E+GfJ)3;36M?IV(Nn22&rzr6uH4-=uOMgml{_4tx-wFIH)==uUte8Y>mH6J-y*YA2Ldm#V9D5yhE|(|c17$TWzkq{5k`nJ}|-T1fa$N{`|oq62q7#*F~izm<aM>J>f&VV}pV1@Dg8k@x)Rpa~LY>B@(iY)T-bjUW`2m5d~Sj*d0~LW)>baK|<r=0+b9qkk`;fXi{(v*#gy`k#wP<Sx*WXLDup<yMYLAeWobkqv}IRBrbvPa}YL{InecN}oA634$Dn;dct?_s%AtJwTg<KL>aB_}?~Pt%FvOfXO2TgB|8Rq;(T}H$WIYRMpvj@T51WxG59Dnj`P1WeJv+<r1t(O(H<BA0(#U0IGvU8dgG4mW_7UGgbUpr`H=OY@kH@Cn3T50L;`*7%fnh5W)V>sv;Vyc`CfIF#5hK5GB=<R3%V32EGiaro{}ZKJY7YA}b!2ZHA8T2yDhYxqwRLb~Irx*fCBJyX*zY$_<SsAL*4g1-0nq3EEI?5s#!`A2Cy0WLcW^x{xLO%hRiycNbSa?9o4cbNSOLTS5SUY7(k8EIcyoWD*8pwNe+TL109^H4NO22kdAvDIs8I{t}{-RFQ!ZH*BbwE1|G@j-W*Q0>82Tp9_r74RQ3xwQ!EqJ-&j%IP59>2YU2&2-5quLH>Z*j1tL7zn)o@EdZY2uBWgKurxjy^(_ZfO_Q<Eo@2q4PnSMYgi-1v?Up&?Kh@NX2T%-&!~#yC@S4pab5~=zfT|nVQzMpep0I1GvsIsfWvveOrm?}Gk45z}4<rw9Ak=+cCIS&*zpOh%jSbz;q=j6$2v-e;uD_vsq-E%|4E7>oNht?|2)azn=U$1=7=o&yDvzfG*~}%DepL4Wh5U&Q6qducNLo>9sJ9G>n^b6%8nmdXQje{cQXM58+gut6kPa~N2VH=;B`f}@EUUI@*AN}2n5jw$w=!xkHmvG}Mv##?_JAcNs4aA1Zw%wm=*dOq#DFZ{rOswjx7`l1=3WTw^lGh#eMie;xfcqmJtj_NS`)<!wWHQBYo{t$NsMoa%9^oG`!EC}2u(eRhXdBDF)>t3KxIlqY)f*Ajg965+2#|^mKaYtl!wD#qa068RM%blCb)t+1AvAcx^4&-r&rYxft|6oT3gdf|0j&Ey>{Ev@LFrS?S^ym$63zE0ykFkcw5n8)J38l8A0o7UmaF%s~(#{6~v{WRL9i>u&RT$0<m?mneS@e$wr`V!t*iVEN?6+;kQ+lAn;mrM_n6xb!I59QT~slKhJmwSpw;Vy?z6(MLJdsTcrV4!&YZ`(t<)4J4X{|5n5Y|KpJNQ+oh*A1`A}bvBG5y3ZDF7B~H|eX`05^*{`EOoV2j4W*R76i9t=UH=`}1UY@oyy^F2pR(3FIW#=+9ZA{<&x|;0OFPr{tOtsVoo_6;f!SC2lA2O6h<Uf}v`xE;YveL?*?Mf;>;;5(o?R!hX6ZjKB&C{!$i6{@hZSn3&J0F$LtDe);XC`{>9k-n0Cfmced9|yaC(JpV4AMjvOR9&#)x;MN37bWD^I<X+AX_c;%}J6D5qAxt#6$A}XbmNuk^?(*QG<62biG3dG>KS4dQRcjeFjoxQSLP(R=qoi-`jxN;%}S$_droM3FmV?Qz2Vby+VyV&wb3{o7$wU?a|{}Cku*tx0dGUjH61_orJnEvHKosSj+`jLf3<}i!!=5?ARt_Uy7C2I%WH;O)6ySxF1LqN!iLMN|mzhbC1>p(XjIN<C2YssWk$dQEcp++BI6!qeG{)ivk$ZP`zLJL0^cZ<f%HsqA6*G326IJfi+`ti0jbMl-`Lt|1mM6ZXB@f{e-`*6?L;oOJDl<&bPnc37N&dr9D~1c3j;k#<~VK3$f~s9YH>VQ>Q}G2BtXO*9hcF1O@cl7|V7W#4=h&exBZm<p0L8mvL+p@1~GV&3GkfXIu$=$L=e`a@T5w4E{(f)0cMYyxqP(SNnuYBXkds6?JH}pm}MVq1F0KS&gVu{MwMbYdw?J+zvX8<LGh853-4t2mj*n<?(m|&EK(i#16`_(O4aPn$P3m@x<a=V^AOJII2;Dr5Y*mD1iZvS6(LKqG$N>_*C|QcO`}rn=_^4QQmK&EH++mDmuE*q)Bx{qweqIon{`}0u4%Ke&2)K`neF|tLP0!CTt`ro(^dQ9+Y&z&VeDJ8ZOwd;R}0&o>~MIe80h(CWse<3S(cRJXEAQ&21#&-p3~<vWMlevOT-f>@A_NfF6q;9t>0?1_6xX6fg2Lz+z1GF(@s~6Rw2NCh{x_P%h2u^~z`WC*XeURPKfWgwUW0@UI8I3v5GYphl=>bDjjkY~}p?vQJH)g6&iVeu>8|XpQ(0sA&~+WpSmhX!WH9zOho}xH;*-wXk<~&sFsUN`MYPzg;pXIwUr8)PUxzDrpF^Xv(U&=uS+lV>u7Nk=$;(3dl!c)*zMjo1Uq*Qwrscdq5q@m2>R7R9_HI^fu_OvvkePQO#`m*7Ljpdpa6F{cbWaZtjfHs)6RJW0hx5*w6Ul&3<qcsZNR7Ef6Q&I;dA|SnT3yxhY2>tI#_)PosJhtqw)`4#|Fiq@MJUlf;5vAkFY)Tv*^`8sZTUAUmd@m9AM)Dp2<jKB>&kURsCNk=112PHsNi8aTy3-H5lfB_!hVCTn2170AUma=~rTcD^}b>inv@rGC76kH$|&lR6LWb4uOTkvgi~{w(iKY(@vRNjPVvt|r{m>V9RcDs*l&_xc6?1cSDNu$0gq*jlgqH=ry>^b;kU;{(UpzO~68G_+NQX+T1MzIAac`RNFL+oi}fDYt<iy%1tpu&5ojlq*UZDSFmoi_0B2T*)*L)%Uamo00`Y(#@=uX4Ts$*nXXg<uFZBJ9bdf+p4U+Xtth+5Up(yp?O|XB6O^e5H;<wqK@ZZNea0Ty6@00PjhHpwa<nZxOX!ymQ7Aj#{2WsN~Vk_R4`y$rcS@~qg)L2=?c<w^pzg;ij~*d6ew$tOZJMT&H`0jIrWdOs+Rmt_@R2_hP&-qR~qV070QNoAOquFUEF-QdRIP(IQ?+``aLYX$qG@2`FMJXwJ^SrvpK_yTE!3SL#j7r=B#lZr&B0docCrlq2D_KfA%7pl4+3zl;wU?|9bc|6ap$$Hfo6O`tWi^1AFx3uGbxO`*c?1toz+mTLv>EoZq@d62-Fnd%f(kUy4xC6{(Xl=dnI1aEwo@*>GsEDun%Vd!A7%J_OHjKgZP^EcMrX<;asJV6cktc(G$mQR8P`syR1(lqvO_OnTsQ_5Lp(u0f;M*Z%bl=+?iue1G=Sbr<@bBgNt)-}BX;!MR>F)*E(4t=C?_t$qlnl%_t`0O+9D2B85LsP-~#mss4z?u#`;#u~N2X|&o8z@tal_-c2)V>U7`*Bpg4PBV<ERNQKRjRb1KM}_@%ROQ9Y7F6d~?=SsJumVCg+KOtcAvp*)m{L1hOR(FqUY)+ZtdJhpk=9D1mQBCAIQy%AdUoc2x_JHLPdA9ft7B;^iCR$oeEP}%<-_ZX8yDY^SC?Sxg}c4a%2W{V`QP82zI}bx@Wx#hY^CKU>1`4&3A)m4vHBYJde7D(BoxiX-_9_&T%F#$evhW$9Y5WSsO<N)V0wFc_2cVzgbJU5mM15ejv)@>3xX|9^}>Dm_t;l|z})Q-knVT)qFZ_3i<8F?F1@lrcv2DGX!tLPAAFFu)u>;%H?bSin$y&p<0ivZk|x7M%;`k5g|SUZ2q^EQTdma(TGShxJt?<)9nf{e(&`}lFjpR%n2XGP`D!=~*`vc$(;fCxly1HhTEouElaYGJt}s=@N8P=%``+){H@n@ASMQtsX3w|qyYtn4Z{>F;sI5=wH#y|)_w`#Hb&r?#+wFe$tNg7VvHiXNM$_K$h<~%a+vQRHdrxIwpnD4XY?_=XG3~_q+3s(H0Jfbl1{uC>e--m3eNzb2Q}qoZ15$6F=t$ljGN4r_qpr-P)$i24_yqvY_rOYGxL4SCgkO8WFG%@5eUHfZamx3*V|{)~aQ_BQ@I}A"
}
_EMBEDDED_TRAINER_SHA256 = {
    "softvq_continuous_online_train_v6": "7da196418589504b8cfda89aae89310c3b4affad5f04ac0e74e88c51cc9b04de",
    "softvq_continuous_online_train_v7": "1774c381ffd922eccfe40fa02bc01ff385546b3b23c71f1b68b6df2fcb8a08a3",
    "softvq_continuous_online_train_v7_1": "90fb68d5678c4c2f13474729039a61a5803041e91a951013025ea683fa280127",
    "softvq_continuous_online_train_v7_2": "5147c4996665c151887909a1182418af6f82d54b11897b0dfd73994e09a518a5"
}

_EMBEDDED_TRAINERS_LOADED = False


def _load_embedded_trainers() -> None:
    global _EMBEDDED_TRAINERS_LOADED
    if _EMBEDDED_TRAINERS_LOADED:
        return
    wrapper_module = sys.modules.get(__name__)
    for module_name, payload in _EMBEDDED_TRAINER_PAYLOADS.items():
        source = zlib.decompress(base64.b85decode(payload.encode("ascii")))
        digest = hashlib.sha256(source).hexdigest()
        expected = _EMBEDDED_TRAINER_SHA256[module_name]
        if digest != expected:
            raise RuntimeError(
                f"embedded trainer checksum mismatch for {module_name}: "
                f"expected {expected}, got {digest}"
            )
        module = types.ModuleType(module_name)
        module.__file__ = str(Path(__file__).resolve()) + f"::<embedded:{module_name}>"
        module.__package__ = ""
        sys.modules[module_name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    # When imported by its historical bare module name, loading the embedded
    # v7 base temporarily occupies the same sys.modules key as this wrapper.
    # v7.1/v7.2 already hold their direct reference to that base, so restore the
    # public wrapper for Personaplex's subsequent import resolution.
    if wrapper_module is not None:
        sys.modules[__name__] = wrapper_module
    _EMBEDDED_TRAINERS_LOADED = True


def _install_personaplex_runtime_contract():
    """Expose v7.2 for Personaplex and require its real dialogue-LLM features.

    The standalone v7.2 trainer keeps ``llm_feat`` optional because its dataset
    does not provide dialogue-LLM states.  End-to-end Personaplex training does
    provide those states, and silently replacing a missing tensor with zeros
    would hide a broken integration.  Keep the pretrained face path unchanged
    at initialization by zeroing the newly connected residual projection.
    """
    _load_embedded_trainers()
    face_cls = sys.modules[
        "softvq_continuous_online_train_v7_2"
    ].CausalSoftVQContinuousTransformer

    def _expand_required_llm(self, llm_feat, audio_feat, face_len):
        if llm_feat is None:
            raise RuntimeError(
                "v7.2 Personaplex face generation requires llm_feat; refusing "
                "to substitute a zero LLM feature."
            )
        if llm_feat.ndim != 3:
            raise ValueError(
                f"llm_feat must have shape [B, T, D], got {tuple(llm_feat.shape)}"
            )
        if llm_feat.shape[0] != audio_feat.shape[0]:
            raise ValueError(
                "LLM/audio batch mismatch: "
                f"llm={llm_feat.shape[0]}, audio={audio_feat.shape[0]}"
            )
        expected_dim = self.llm_proj.in_features
        if llm_feat.shape[-1] != expected_dim:
            raise ValueError(
                f"LLM feature dim mismatch: got {llm_feat.shape[-1]}, "
                f"expected {expected_dim}"
            )
        llm_rep = llm_feat.repeat_interleave(2, dim=1)
        if llm_rep.shape[1] < face_len:
            raise RuntimeError(
                "LLM feature coverage is too short for face generation: "
                f"expanded_llm={llm_rep.shape[1]}, face_len={face_len}"
            )
        return llm_rep[:, :face_len]

    face_cls._expand_llm = _expand_required_llm

    original_decode_motion = face_cls._decode_motion

    def _decode_motion_in_projection_dtype(
        self,
        audio_h,
        prev_motion,
        pos_offset=0,
        blink_cond=None,
        partner_h=None,
        role_cond=None,
    ):
        # Scheduled sampling can feed a float32 prediction back as prev_motion
        # while FSDP keeps the face projection in bf16.  torch.cat promotes the
        # whole input to float32, which Linear cannot multiply by bf16 weights.
        # Cast every floating projection input explicitly; ``Tensor.to`` keeps
        # the autograd edge intact.
        projection_dtype = self.motion_proj.weight.dtype
        audio_h = audio_h.to(dtype=projection_dtype)
        prev_motion = prev_motion.to(dtype=projection_dtype)
        if partner_h is not None:
            partner_h = partner_h.to(dtype=projection_dtype)
        return original_decode_motion(
            self,
            audio_h,
            prev_motion,
            pos_offset,
            blink_cond=blink_cond,
            partner_h=partner_h,
            role_cond=role_cond,
        )

    face_cls._decode_motion = _decode_motion_in_projection_dtype

    original_init = face_cls.__init__

    def _zero_init_llm_projection(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        torch.nn.init.zeros_(self.llm_proj.weight)
        if self.llm_proj.bias is not None:
            torch.nn.init.zeros_(self.llm_proj.bias)

    face_cls.__init__ = _zero_init_llm_projection
    return face_cls


# ``lm2.py`` imports this symbol directly when face_module_version >= 7.
CausalSoftVQContinuousTransformer = _install_personaplex_runtime_contract()

# END GENERATED STANDALONE TRAINER PAYLOAD

# ===========================================================================
# PORTABLE PATH CONFIG -- EDIT ONLY THIS BLOCK WHEN MOVING TO ANOTHER SERVER.
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
UNILS_DATA_ROOT = Path(
    "/home6/duplex/dataset/unils/SeamlessInteractionTalk/flame56"
)
UNILS_MIMI_ROOT = Path("/home6/duplex/dataset/mimi_emb/unils")
CODEC_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/ARTalkCodecMimi54_MimiFlame54/Jul21_0944_empng/checkpoints/iter_50000.pt"
)
MOTION_STATS_JSON = Path(
    "/home6/duplex/dataset/artalk_mimi54_unils_stats.json"
)
EYELID_PROBE = PROJECT_ROOT / "assets/eyelid_probe.pt"
LENGTH_MISMATCH_CSV = Path(
    "/home6/duplex/dataset/artalk_mimi54_length_mismatches.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs/SoftVQ_v7_2_unils_partner_no_neck_acc_e2e"
# ===========================================================================


# Every value below matches the evaluated v7.2 e125 checkpoint.  Paths are
# filled from PORTABLE PATH CONFIG so a server move needs one edit location.
PAPER_CONFIG = {
    "codec_ckpt": str(CODEC_CHECKPOINT),
    "stats_path": str(MOTION_STATS_JSON),
    "output_dir": str(OUTPUT_DIR),
    "resume": "",
    "dualtalk_root": "/home6/duplex/dataset/dualtalk",
    "mimi_root": "/home6/duplex/dataset/mimi_emb",
    "ami_bc_root": "/home6/duplex/dataset/ami_flame/bc",
    "ami_bc_mimi_root": "/home6/duplex/dataset/mimi_emb",
    "ami_ut_root": "/home6/duplex/dataset/ami_flame/ut",
    "ami_ut_mimi_root": "/home6/duplex/dataset/mimi_emb",
    "unils_root": str(UNILS_DATA_ROOT),
    "unils_mimi_root": str(UNILS_MIMI_ROOT),
    "sources": "unils",
    "length_mismatch_csv": str(LENGTH_MISMATCH_CSV),
    "epochs": 800,
    "batch_size": 1024,
    "num_workers": 4,
    "clip_length": 100,
    "stride": 50,
    "lr": 1.0e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "precision": "bf16",
    "hidden_dim": 512,
    "layers": 6,
    "heads": 8,
    "code_dim": 32,
    "codec_hidden_dim": 512,
    "codec_depth": 8,
    "codec_heads": 8,
    "motion_weight": 1.0,
    "prior_weight": 0.5,
    "z_weight": 0.2,
    "z_bce_weight": 0.1,
    "jaw_weight": 2.0,
    "vel_weight": 0.5,
    "reg_weight": 0.02,
    "gate_weight": 1.0,
    "gate_target_expr": 0.45,
    "gate_target_jaw": 0.25,
    "gate_target_neck": 0.65,
    "gate_loss_expr_weight": 4.0,
    "gate_loss_jaw_weight": 4.0,
    "gate_loss_neck_weight": 1.0,
    "prior_warmup_frames": 10,
    "lookahead_frames": 4,
    "lookahead_weight": 0.5,
    "token_vocab": 2048,
    "mtp_horizon_gamma": 0.8,
    "la_temp": 0.5,
    "spec_feat_weight": 0.1,
    "chunk_frames": 4,
    "eos_prob": 0.25,
    "blink_weight": 0.1,
    "blink_pos_weight": 10.0,
    "blink_thresh": 0.6,
    "blink_ap_weight": 0.1,
    "blink_ap_boost": 9.0,
    "blink_temp": 1.0,
    "blink_refractory": 12,
    "eyelid_probe": str(EYELID_PROBE),
    "prev_noise_std": 0.015,
    "ss_prob": 0.25,
    "ss_passes": 2,
    "ss_ramp_epochs": 100,
    "ss_keep_head_frames": 0,
    "spec_topk": 4,
    "ar_eval_frames": 400,
    "val_speculative": True,
    "val_batches": 16,
    "val_every_epochs": 25,
    "save_val_samples": 4,
    "save_every_epochs": 25,
    "stream_context_frames": 50,
    "max_train_batches": 0,
    "seed": 42,
    "wandb": True,
    "wandb_project": "NIPS_duplex_SoftVQ_Continuous",
    "wandb_run_name": "softvq_v7_2_unils_partner_no_neck_acc_e2e",
    # Derived v7.1/v7.2 values recorded in the checkpoint.
    "partner_layers": 2,
    "use_vap": False,
    "vap_weight": 0.0,
    "v72_robustness_start_epochs": 10,
    "v72_partner_ema_alpha": 0.75,
    "v72_partner_drop_prob": 0.15,
    "v72_partner_drop_start_epochs": 10,
    "v72_partner_drop_ramp_epochs": 20,
    "v72_neck_acc_weight": 0.0,
    "v72_jaw_quiet_acc_weight": 0.5,
    "v72_dynamics_start_epochs": 10,
    "v72_dynamics_ramp_epochs": 100,
    "v72_acc_excess_ratio": 1.25,
    "v72_acc_excess_margin": 1.0e-4,
}


V72_ENV = {
    "V72_ROBUSTNESS_START_EPOCHS": "10",
    "V72_PARTNER_EMA_ALPHA": "0.75",
    "V72_PARTNER_DROP_PROB": "0.15",
    "V72_PARTNER_DROP_START_EPOCHS": "10",
    "V72_PARTNER_DROP_RAMP_EPOCHS": "20",
    "V72_NECK_ACC_WEIGHT": "0.0",
    "V72_JAW_QUIET_ACC_WEIGHT": "0.5",
    "V72_DYNAMICS_START_EPOCHS": "10",
    "V72_DYNAMICS_RAMP_EPOCHS": "100",
    "V72_ACC_EXCESS_RATIO": "1.25",
    "V72_ACC_EXCESS_MARGIN": "0.0001",
}


# Arguments understood by the inherited v7 parser.  Derived partner/v7.2
# values are applied by v7.2.parse_args from the locked environment above.
PARSER_FIELDS = (
    "codec_ckpt", "stats_path", "output_dir", "resume",
    "dualtalk_root", "mimi_root", "ami_bc_root", "ami_bc_mimi_root",
    "ami_ut_root", "ami_ut_mimi_root", "unils_root", "unils_mimi_root",
    "sources", "length_mismatch_csv", "epochs", "batch_size", "num_workers",
    "clip_length", "stride", "lr", "weight_decay", "grad_clip", "precision",
    "hidden_dim", "layers", "heads", "code_dim", "codec_hidden_dim",
    "codec_depth", "codec_heads", "motion_weight", "prior_weight", "z_weight",
    "z_bce_weight", "jaw_weight", "vel_weight", "reg_weight", "gate_weight",
    "gate_target_expr", "gate_target_jaw", "gate_target_neck",
    "gate_loss_expr_weight", "gate_loss_jaw_weight", "gate_loss_neck_weight",
    "prior_warmup_frames", "lookahead_frames", "lookahead_weight",
    "token_vocab", "mtp_horizon_gamma", "la_temp", "spec_feat_weight",
    "chunk_frames", "eos_prob", "blink_weight", "blink_pos_weight",
    "blink_thresh", "blink_ap_weight", "blink_ap_boost", "blink_temp",
    "blink_refractory", "eyelid_probe", "prev_noise_std", "ss_prob",
    "ss_passes", "ss_ramp_epochs", "ss_keep_head_frames", "spec_topk",
    "ar_eval_frames", "val_batches", "val_every_epochs", "save_val_samples",
    "save_every_epochs", "stream_context_frames", "max_train_batches", "seed",
    "wandb_project", "wandb_run_name",
)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _locked_trainer_argv(batch_size: int) -> list[str]:
    values = dict(PAPER_CONFIG)
    values["batch_size"] = batch_size
    result = [str(Path(__file__).resolve())]
    for name in PARSER_FIELDS:
        value = values[name]
        if name == "resume" and not value:
            continue
        result.extend([_flag(name), str(value)])
    result.append("--wandb" if values["wandb"] else "--no-wandb")
    result.append(
        "--val-speculative"
        if values["val_speculative"]
        else "--no-val-speculative"
    )
    return result


def _load_semantic_tokens(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tokens = payload.get("audio_tokens") if isinstance(payload, dict) else payload
    if not torch.is_tensor(tokens) or tokens.ndim != 2 or tokens.shape[0] < 1:
        raise ValueError(f"invalid Mimi token sidecar: {path}")
    return tokens[0].long()


def preflight_e2e_files() -> None:
    required_files = {
        "codec checkpoint": CODEC_CHECKPOINT,
        "motion statistics": MOTION_STATS_JSON,
        "eyelid probe": EYELID_PROBE,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "v7.2 e2e preflight missing required files:\n" + "\n".join(missing)
        )

    with MOTION_STATS_JSON.open() as handle:
        stats = json.load(handle)
    if not isinstance(stats, dict) or not stats:
        raise ValueError(f"invalid or empty motion statistics: {MOTION_STATS_JSON}")

    counts = {}
    for split in ("train", "val"):
        motion_dir = UNILS_DATA_ROOT / split
        mimi_dir = UNILS_MIMI_ROOT / split
        if not motion_dir.is_dir() or not mimi_dir.is_dir():
            raise FileNotFoundError(
                f"missing UniLS {split} input: {motion_dir} or {mimi_dir}"
            )
        motion_stems = {
            path.stem
            for path in motion_dir.glob("*.npy")
            if not path.name.endswith(".fbank.npy")
        }
        latent_stems = {
            path.name[: -len("_latent.pt")]
            for path in mimi_dir.glob("*_latent.pt")
        }
        token_stems = {
            path.name[: -len("_token.pt")]
            for path in mimi_dir.glob("*_token.pt")
        }
        if not motion_stems:
            raise FileNotFoundError(f"no motion files in {motion_dir}")
        missing_latents = sorted(motion_stems - latent_stems)
        missing_tokens = sorted((motion_stems & latent_stems) - token_stems)
        if missing_latents:
            raise FileNotFoundError(
                f"{split}: {len(missing_latents)} motions lack Mimi latent; "
                f"first={missing_latents[:5]}"
            )
        if missing_tokens:
            raise FileNotFoundError(
                f"{split}: {len(missing_tokens)} Mimi latents lack MTP tokens; "
                f"first={missing_tokens[:5]}"
            )
        sample_stem = next(iter(sorted(motion_stems)))
        semantic = _load_semantic_tokens(mimi_dir / f"{sample_stem}_token.pt")
        if bool((semantic < 0).any()) or bool(
            (semantic >= PAPER_CONFIG["token_vocab"]).any()
        ):
            raise ValueError(
                f"semantic token outside [0,{PAPER_CONFIG['token_vocab']}) "
                f"in {mimi_dir / f'{sample_stem}_token.pt'}"
            )
        counts[split] = len(motion_stems)
    print(
        "[v7.2-e2e-preflight] locked paper config verified "
        f"(train={counts['train']}, val={counts['val']})"
    )


def _frontend_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked v7.2 UniLS paper run; only batch size is mutable"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(PAPER_CONFIG["batch_size"]),
        help="per-rank batch size; the only permitted paper-config override",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    return args


def _verify_parsed_config(args, batch_size: int) -> None:
    expected = dict(PAPER_CONFIG)
    expected["batch_size"] = batch_size
    mismatches = {}
    for name, value in expected.items():
        if not hasattr(args, name):
            mismatches[name] = (value, "<missing>")
        elif getattr(args, name) != value:
            mismatches[name] = (value, getattr(args, name))
    if mismatches:
        detail = ", ".join(
            f"{name}: expected {expected_value!r}, got {actual!r}"
            for name, (expected_value, actual) in sorted(mismatches.items())
        )
        raise ValueError(f"v7.2 e2e locked-config mismatch: {detail}")


def main() -> None:
    frontend = _frontend_args()
    preflight_e2e_files()

    # Do not let inherited shell/environment defaults silently change the
    # evaluated v7.2 configuration.
    for name, value in V72_ENV.items():
        os.environ[name] = value
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _load_embedded_trainers()
    v72 = sys.modules["softvq_continuous_online_train_v7_2"]

    original_parse_args = v72.parse_args

    def locked_parse_args():
        args = original_parse_args()
        _verify_parsed_config(args, frontend.batch_size)
        return args

    v72.parse_args = locked_parse_args
    sys.argv = _locked_trainer_argv(frontend.batch_size)
    v72.main()


if __name__ == "__main__":
    main()
