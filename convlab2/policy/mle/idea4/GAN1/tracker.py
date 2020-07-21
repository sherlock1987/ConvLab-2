from collections import defaultdict
dicc = {'test_eval-': [2.6845746845746845, 3.6993623660290327, 2.718898385565052, 3.3230226563559895, 2.8129154795821463, 3.398860398860399, 2.6848460181793516, 3.3767467100800435, 2.5707502374169042, 3.1930538597205262, 2.6358703025369694, 3.3816307149640483, 2.758784425451092, 3.1850495183828516, 2.745217745217745, 3.2611585944919277, 2.721069054402388, 3.514855514855515, 2.904490571157238, 3.265771265771266, 2.722290055623389, 3.225342558675892, 2.6544566544566544, 3.9739519739519737, 2.6105006105006106, 3.436168769502103, 2.5925925925925926, 3.4612671279337945, 2.758784425451092, 3.3228869895536564, 2.5925925925925926, 3.3498846832180167, 2.6661239994573327, 3.51309184642518, 2.621625288291955, 3.176502509835843, 2.581739248405915, 3.429928096594763, 2.907610907610908, 3.0872337539004207, 2.643060643060643, 3.305114638447972, 2.6706010039343373, 3.202279202279202, 2.798670465337132, 3.411613078279745, 2.6258309591642925, 3.2157102157102155, 2.62650929317596, 3.3064713064713063], 'test_eval*': [0.9091032424365758, 1.1792158458825126, 0.9116809116809117, 0.9701533034866369, 0.9587572920906254, 0.9906389906389906, 0.7969067969067969, 1.1202007868674535, 0.8346221679555013, 0.9597069597069597, 0.7992131325464659, 1.0363587030253696, 0.919006919006919, 1.0183150183150182, 0.8891602224935559, 1.0407000407000406, 0.9148012481345814, 1.0714964048297382, 0.901370234703568, 1.111653778320445, 0.8534798534798534, 1.0657983991317324, 0.807353140686474, 1.0004070004070005, 0.7883597883597884, 1.1054131054131053, 0.8154931488264822, 0.9792429792429792, 0.8430335097001763, 1.1233211233211233, 0.7172703839370506, 1.0302536969203635, 0.7936507936507936, 1.139465472798806, 0.7321937321937322, 1.0400217066883735, 0.73992673992674, 1.0077330077330078, 0.960385293718627, 0.9621489621489622, 0.7817121150454484, 0.9550942884276218, 0.7418260751594085, 0.9740876407543074, 0.8708452041785375, 1.0062406729073397, 0.781305114638448, 1.0487043820377153, 0.8590421923755257, 1.021978021978022], 'average_act': [2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804, 2.5471442138108804], 'd_loss': [0.2536856320518509, 0.12483318821231296, 0.20458512414919325, 0.08707765194213583, 0.1922451882483881, 0.09071027555930271, 0.18312965478343668, 0.09286831575539552, 0.1838242811800301, 0.08681884127626852, 0.17533204786334286, 0.09395726485218076, 0.1784128195752925, 0.10028235595382497, 0.1740967889311063, 0.08847399148036336, 0.18681807110256882, 0.08380032455053538, 0.16296308300990364, 0.0886140475949424, 0.16654939480380693, 0.09649799628217448, 0.1707904347699814, 0.03694542101811231, 0.17014610748813414, 0.08624741270698344, 0.1793590556286776, 0.08255563481225582, 0.17109451072533222, 0.08767938137951142, 0.16395052741555607, 0.08377107141599575, 0.16620886517870972, 0.0824013359072016, 0.16250333434548234, 0.09380834308422029, 0.1665426560125935, 0.10430349075443451, 0.17199772893142054, 0.09858894237316604, 0.1741337157297783, 0.0883377802439741, 0.15219289888211382, 0.0879484598771998, 0.16461182568351954, 0.08372473088599836, 0.17077468688286992, 0.09136235313371324, 0.16986121394829337, 0.08935137531002892], 'd_eval': [0.01095050544518542, 0.0352440045740434, 0.07057606549594994, 0.07722257184871664, 0.25105364819703235, 0.133904239058683, 0.2542490412312433, 0.08307594619584262, 0.14495335956500444, 0.02313797343760279, 0.07031386163335741, 0.12339203100410297, 0.20362282465917508, 0.26641204446558536, 0.3551571326129986, 0.2670068522277071, 0.14301970732420363, 0.25842665798014486, 0.1776983308369102, 0.14572528045412617, 0.13613063962597402, 0.1906339978379572, 0.4763093402394575, 0.19792301559605288, 0.15145547286840083, 0.09674547935579215, 0.06914954541378547, 0.0738593835489844, 0.2167039145508558, 0.27228588411609017, 0.2606478859396542, 0.25806033219595087, 0.2501007467692965, 0.18161334240789284, 0.06542140834284996, 0.25685103806297466, 0.16528274181830244, 0.13133559784762738, 0.18335112184630678, 0.3022089619611599, 0.1982499583542568, 0.17873927779281787, 0.1753160430163872, 0.3041146500500163, 0.024820438796108773, 0.15169165524966666, 0.2796981861646161, 0.05504400401375109, 0.037395843800466705, 0.148973575289837], 'd_pos': [0.6976127320954907, 0.8520486645224444, 0.7393715341959335, 0.8747099767981439, 0.7124654590113602, 0.87138956746659, 0.7911499004746593, 0.8904542840713053, 0.762816553428042, 0.8934378629500581, 0.777296628518404, 0.8735251798561151, 0.7738945449019906, 0.8485334889829271, 0.7239623514889677, 0.8671308384102118, 0.7354260089686099, 0.8754160034727246, 0.7707249923524013, 0.8783297336213103, 0.7627664468639779, 0.8668026814339843, 0.7221189591078067, 0.9392068066419651, 0.7687519452225334, 0.8866810655147588, 0.7644781666150511, 0.8933584141224136, 0.7664579196578586, 0.8693765369593519, 0.7643995098039216, 0.8767083454492585, 0.7692069392812887, 0.8881068658431486, 0.7869309383710127, 0.8668029794070395, 0.7701418108150226, 0.8653428033048268, 0.7675350701402806, 0.8550261475886113, 0.761742892459827, 0.8780061215566244, 0.7932198189906428, 0.8676129779837776, 0.7902416780665754, 0.8917525773195877, 0.7544188956671359, 0.8893395133256083, 0.7839411584431505, 0.8844633182017865], 'd_neg': [0.6342643158058979, 0.8380646063487625, 0.7164202094886013, 0.9090777262180975, 0.7731040835124348, 0.904296594338267, 0.7178073801868015, 0.8836975273145485, 0.7461395923409512, 0.8975029036004646, 0.7658521497061552, 0.9024460431654676, 0.7608266220939067, 0.9098205165620896, 0.833667643882117, 0.9289237017696548, 0.7917117674346683, 0.9302561134423383, 0.8178342000611808, 0.9144708423326134, 0.8188928078515565, 0.9078985718449432, 0.838909541511772, 0.97667078358721, 0.7998755057578587, 0.9105831533477322, 0.7881697119851347, 0.9107220373317899, 0.8148770429204215, 0.9266599161001012, 0.8262867647058824, 0.9303576621110788, 0.8241945477075588, 0.919994254524562, 0.8101579436358005, 0.9177742076821965, 0.8120617110799438, 0.8918683867227134, 0.8037613688916294, 0.9174898314933179, 0.8104140914709518, 0.9170674828742166, 0.8358643963798128, 0.9339513325608343, 0.8116735066119471, 0.9206758304696449, 0.8338808071328015, 0.9019409038238703, 0.8034017775053631, 0.9115536681798213]}

for keys, value in dicc.items():
    print(keys)
    print(value)