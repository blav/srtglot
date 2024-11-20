from contextlib import ExitStack
from pathlib import Path
import datetime
from unittest.mock import MagicMock, patch
from srtglot.sentence import (
    collect_sentences,
    token_counter,
    sentences_batcher,
)
from srtglot.parser import parse
from srtglot.model import Sentence, Subtitle, Multiline
from fixtures import srt_file


def test_token_counter_should_count_token():
    sentence = Sentence(
        blocks=[
            Subtitle(
                start=datetime.time(0, 0, 12, 178000),
                end=datetime.time(0, 0, 14, 848000),
                text=[
                    Multiline(lines=["As the first century"]),
                    Multiline(lines=[""]),
                    Multiline(lines=["of the Targaryen dynasty."]),
                ],
                soup=None,
            )
        ]
    )

    assert token_counter("gpt-4o")(sentence) == 12


def test_should_batch_sentences():
    with ExitStack() as stack:
        counter = MagicMock()
        counter.side_effect = [2, 1, 1, 3, 2, 1]
        token_counter = stack.enter_context(
            patch("srtglot.sentence.token_counter")
        )
        token_counter.return_value = counter

        s1 = MagicMock()
        s2 = MagicMock()
        s3 = MagicMock()
        s4 = MagicMock()
        s5 = MagicMock()
        s6 = MagicMock()
        sentences = MagicMock()
        sentences.__iter__.return_value = [s1, s2, s3, s4, s5, s6]
        batcher = sentences_batcher("gpt-4o", 4)
        assert [*batcher(sentences)] == [
            [s1, s2, s3],
            [s4],
            [s5, s6],
        ]


def test_should_return_sentences(srt_file: Path):
    sentences = [*collect_sentences(parse(srt_file))]
    assert len(sentences) == 405
    assert (
        str(sentences[0])
        == "As the first century of the Targaryen dynasty came to a close, the health of the Old King, Jaehaerys, was failing."
    )
    assert (
        str(sentences[1])
        == "In those days, House Targaryen stood at the height of its strength with 10 adult dragons under its yoke."
    )
    assert str(sentences[2]) == "No power in the world could stand against it."
    assert (
        str(sentences[3])
        == "King Jaehaerys reigned over nearly 60 years of peace and prosperity."
    )
    assert (
        str(sentences[4])
        == "But tragedy had claimed both his sons, leaving his succession in doubt."
    )
    assert (
        str(sentences[5])
        == "So, in the year 101, the Old King called a Great Council to choose an heir."
    )
    assert str(sentences[6]) == "Over a thousand lords made the journey to Harrenhal."
    assert (
        str(sentences[7])
        == "Fourteen succession claims were heard but only two were truly considered: Princess Rhaenys Targaryen, the King's eldest descendant, and her younger cousin, Prince Viserys Targaryen, the King's eldest male descendant."
    )
    assert (
        str(sentences[8])
        == "It is declared by all lords paramount and lords vassal of the Seven Kingdoms, that Prince Viserys Targaryen be made Prince of Dragonstone."
    )
    assert str(sentences[9]) == "Rhaenys, a woman, would not inherit the Iron Throne."
    assert str(sentences[10]) == "The lords instead chose Viserys, my father."
    assert (
        str(sentences[11])
        == "Jaehaerys called the Great Council to prevent a war being fought over his succession."
    )
    assert str(sentences[12]) == "For he knew the cold truth."
    assert (
        str(sentences[13])
        == "The only thing that could tear down the House of the Dragon, was itself."
    )
    assert str(sentences[14]) == "Dohaerās, Syrax! Umbās."
    assert str(sentences[15]) == "Rybās! Welcome back, Princess."
    assert str(sentences[16]) == "I trust your ride was pleasant."
    assert str(sentences[17]) == "Try not to look too relieved, ser."
    assert str(sentences[18]) == "I am relieved."
    assert (
        str(sentences[19])
        == "Every time that golden beast brings you back unspoiled, it saves my head from a spike."
    )
    assert str(sentences[20]) == "Syrax is growing quickly."
    assert str(sentences[21]) == "She'll soon be as large as Caraxes."
    assert str(sentences[22]) == "That's almost large enough to saddle two."
    assert (
        str(sentences[23]) == "I believe I'm quite content as a spectator, thank you."
    )
    assert str(sentences[24]) == "Dohaerās."
    assert str(sentences[25]) == "Naejot! Rhaenyra."
    assert (
        str(sentences[26])
        == "You know I don't like you to go flying while I'm in this condition."
    )
    assert (
        str(sentences[27])
        == "You don't like me to go flying while you're in any condition."
    )
    assert str(sentences[28]) == "-Your Grace. -Good morrow, Alicent."
    assert (
        str(sentences[29])
        == "Did you sleep? -I slept. -How long? I don't need mothering, Rhaenyra."
    )
    assert (
        str(sentences[30])
        == "Well, here you are, surrounded by attendants, all focused on the babe."
    )
    assert str(sentences[31]) == "Someone has to attend to you."
    assert str(sentences[32]) == "You will lie in this bed soon enough, Rhaenyra."
    assert str(sentences[33]) == "This discomfort is how we serve the realm."
    assert (
        str(sentences[34])
        == "I'd rather serve as a knight and ride to battle and glory."
    )
    assert str(sentences[35]) == "We have royal wombs, you and I."
    assert str(sentences[36]) == "The childbed is our battlefield."
    assert str(sentences[37]) == "We must learn to face it with a stiff lip."
    assert str(sentences[38]) == "Now take a bath."
    assert str(sentences[39]) == "You stink of dragon."
    assert (
        str(sentences[40])
        == 'So, I said to him, "Well, I believe you might be looking up the wrong end." My lords.'
    )
    assert (
        str(sentences[41])
        == 'The growing alliance among the Free Cities has taken to styling itself "the Triarchy." They have massed on Bloodstone and are presently ridding the Stepstones of its pirate infestation.'
    )
    assert (
        str(sentences[42])
        == "Well, that sounds suspiciously like good news, Lord Corlys."
    )
    assert (
        str(sentences[43])
        == "A man called Craghas Drahar has styled himself the prince-admiral of this Triarchy."
    )
    assert (
        str(sentences[44])
        == 'They call him "The Crabfeeder" due to his inventive methods of punishing his enemies.'
    )
    assert (
        str(sentences[45])
        == "And are we meant to weep for dead pirates? -No, Your Grace. -Rhaenyra, you're late."
    )
    assert str(sentences[46]) == "King's cupbearer must not be late."
    assert (
        str(sentences[47]) == "-Leaves people wanting for cups. -I was visiting Mother."
    )
    assert (
        str(sentences[48])
        == "On dragonback? Hey, Your Grace, at Prince Daemon's urging, the crown has invested significant capital in the re-training and re-equipping of his City Watch."
    )
    assert (
        str(sentences[49])
        == "I thought you might urge your brother to fill his seat on the council and provide an assessment of his progress as commander of the Watch."
    )
    assert (
        str(sentences[50])
        == "Do you think Daemon is distracted by his present tasks? And that his thoughts and energies are occupied? Well, one would hope so, considering the associated costs."
    )
    assert (
        str(sentences[51])
        == "Then let us all consider your gold well-invested, Lord Beesbury."
    )
    assert (
        str(sentences[52])
        == "I would urge that you not allow this Triarchy much latitude in the Stepstones, Your Grace."
    )
    assert (
        str(sentences[53])
        == "If those shipping lanes should fall, it will beggar our ports."
    )
    assert (
        str(sentences[54])
        == "The crown has heard your report, Lord Corlys, and takes it under advisement."
    )
    assert (
        str(sentences[55])
        == "Shall we discuss the Heir's Tournament, Your Grace? I would be delighted."
    )
    assert (
        str(sentences[56])
        == "Will the maesters' name day prediction hold, Mellos? You must understand that these things are mere estimations, my King, but we have all been poring over the moon charts, and we feel that our forecast is as accurate as it can be."
    )
    assert str(sentences[57]) == "The cost of the tournament is not negligible."
    assert (
        str(sentences[58])
        == "Perhaps we might delay until the child is in hand? Most of the lords and knights are certainly on their way to King's Landing already. To turn them back now-- The tourney will take the better part of a week."
    )
    assert (
        str(sentences[59])
        == "Before the games are over, my son will be born, and the whole realm will celebrate."
    )
    assert str(sentences[60]) == "We have no way of predicting the sex of the child."
    assert (
        str(sentences[61])
        == "Of course, no maester's capable of rendering an opinion free of conditions, are they now? There's a boy in the Queen's belly."
    )
    assert str(sentences[62]) == "I know it."
    assert (
        str(sentences[63])
        == "And my heir will soon put all of this damnable hand-wringing to rest himself."
    )
    assert (
        str(sentences[64]) == "He passed through the Red Keep's gates at first light."
    )
    assert str(sentences[65]) == "Does my father know he's here? -No. -Good."
    assert str(sentences[66]) == "Gods be good."
    assert str(sentences[67]) == "It's all right, ser."
    assert str(sentences[68]) == "Aye."
    assert str(sentences[69]) == "I bought you something."
    assert str(sentences[70]) == "Do you know what it is? It's Valyrian steel."
    assert str(sentences[71]) == "Like Dark Sister."
    assert str(sentences[72]) == "Turn around."
    assert (
        str(sentences[73]) == "Now, you and I both own a small piece of our ancestry."
    )
    assert str(sentences[74]) == "Did you read it? Of course, I read it."
    assert (
        str(sentences[75])
        == "When Princess Nymeria arrived in Dorne, who did she take to husband? A man."
    )
    assert str(sentences[76]) == "What was his name? Lord Something."
    assert (
        str(sentences[77])
        == 'If you answer with "Lord Something," Septa Marlow will be furious.'
    )
    assert str(sentences[78]) == "She's funny when she's furious."
    assert str(sentences[79]) == "You're always like this when you're worried."
    assert str(sentences[80]) == "Like what? Disagreeable."
    assert (
        str(sentences[81])
        == "You're worried your father is about to overshadow you with a son."
    )
    assert str(sentences[82]) == "I only worry for my mother."
    assert str(sentences[83]) == "I hope for my father that he gets a son."
    assert str(sentences[84]) == "As long as I can recall, it's all he's wanted."
    assert (
        str(sentences[85])
        == "You want him to have a son? I want to fly with you on dragonback, see the great wonders across the Narrow Sea, and eat only cake."
    )
    assert str(sentences[86]) == "-I'm being serious. -I never jest about cake."
    assert (
        str(sentences[87])
        == "You aren't worried about your position? I like this position. It's quite comfortable."
    )
    assert str(sentences[88]) == "-Where are you going? -Home. The hour has grown late."
    assert (
        str(sentences[89])
        == "Princess Nymeria led her Rhoynar across the Narrow Sea on 10,000 ships to flee their Valyrian pursuers."
    )
    assert (
        str(sentences[90])
        == "She took Lord Mors Martell of Dorne to husband and burned her own fleet off Sunspear to show her people that they were finished running."
    )
    assert str(sentences[91]) == "-What are you doing? -So you remember."
    assert str(sentences[92]) == "-If the Septa sees this book, then-- -Fuck the Septa."
    assert (
        str(sentences[93])
        == "Rhaenyra! Is it healing? It has grown slightly, Your Grace."
    )
    assert (
        str(sentences[94])
        == "Can you say yet what it is? We've sent inquiries to the Citadel."
    )
    assert str(sentences[95]) == "They are searching the texts for similar cases."
    assert str(sentences[96]) == "It's a small cut from sitting the throne."
    assert str(sentences[97]) == "It's nothing."
    assert (
        str(sentences[98])
        == "The King has been under heavy stresses preparing for the birth."
    )
    assert str(sentences[99]) == "Bad humors of the mind can adversely affect the body."
    assert str(sentences[100]) == "Whatever it is, it needs to be kept quiet."
    assert str(sentences[101]) == "We should leech it again, maester."
    assert str(sentences[102]) == "It's a wound that refuses to heal, Grand Maester."
    assert (
        str(sentences[103])
        == "Might I suggest cauterization? Cauterization would be a wise course of treatment, Your Grace."
    )
    assert str(sentences[104]) == "-It will be painful. -Fine."
    assert str(sentences[105]) == "Fine."
    assert (
        str(sentences[106])
        == "You spend more time in that bath than I do on the throne."
    )
    assert (
        str(sentences[107]) == "This is the only place I can find comfort these days."
    )
    assert str(sentences[108]) == "It's tepid."
    assert str(sentences[109]) == "It's as warm as the maesters will allow."
    assert (
        str(sentences[110])
        == "Don't they know dragons prefer heat? After this miserable pregnancy, I wouldn't be surprised if I hatched an actual dragon."
    )
    assert str(sentences[111]) == "Then he will be loved and cherished."
    assert (
        str(sentences[112])
        == "Rhaenyra has already declared that she is to have a sister."
    )
    assert str(sentences[113]) == "Really? -She even named her. -Dare I ask? Visenya."
    assert (
        str(sentences[114])
        == "She chose a dragon's egg for the cradle that she said reminded her of Vhagar."
    )
    assert str(sentences[115]) == "Gods be good."
    assert str(sentences[116]) == "This family already has its Visenya."
    assert (
        str(sentences[117])
        == "Has there been any word from your dear brother? Not since I named him Commander of the City Watch."
    )
    assert str(sentences[118]) == "I'm sure he will reemerge for the tourney."
    assert str(sentences[119]) == "He could never stay away from the lists."
    assert (
        str(sentences[120])
        == "The tourney to celebrate the firstborn son that we presently do not have."
    )

    assert (
        str(sentences[121])
        == "You do understand nothing will cause the babe to grow a cock if it does not already possess one? This child is a boy, Aemma."
    )
    assert str(sentences[122]) == "I'm certain of it."
    assert str(sentences[123]) == "I've never been more certain of anything."
    assert str(sentences[124]) == "The dream."
    assert str(sentences[125]) == "It was clearer than a memory."
    assert str(sentences[126]) == "Our son was born wearing Aegon's iron crown."
    assert (
        str(sentences[127])
        == "When I heard the sound of thundering hooves, splintering shields, and ringing swords, and I placed our son upon the Iron Throne as the bells of the Grand Sept tolled and all the dragons roared as one."
    )
    assert (
        str(sentences[128])
        == "Born wearing a crown? Gods spare me, birth is unpleasant enough as it is."
    )
    assert str(sentences[129]) == "This is the last time, Viserys."
    assert (
        str(sentences[130])
        == "I've lost one babe in the cradle, had two stillbirths, and two pregnancies ended well before their term."
    )
    assert str(sentences[131]) == "That's five in twice as many years."
    assert (
        str(sentences[132])
        == "I know it is my duty to provide you an heir, and I'm sorry if I have failed you in that. I am."
    )
    assert str(sentences[133]) == "But I've mourned all the dead children I can."
    assert (
        str(sentences[134])
        == "Commander on the floor! When I took command of the Watch, you were stray mongrels, starving and undisciplined."
    )
    assert str(sentences[135]) == "Now, you're a pack of hounds."
    assert str(sentences[136]) == "You're sated and honed for the hunt."
    assert str(sentences[137]) == "My brother's city has fallen into squalor."
    assert str(sentences[138]) == "Crime of every breed has been allowed to thrive."
    assert str(sentences[139]) == "No longer."
    assert (
        str(sentences[140])
        == "Beginning tonight, King's Landing will learn to fear the color gold."
    )
    assert (
        str(sentences[141])
        == "Get up! Raper! No! No! No! No! No! Thief! No! Murderer! It was an unprecedented roundup of criminals of every ilk."
    )
    assert (
        str(sentences[142])
        == "Your brother made a public show of it, meting out the summary judgments himself."
    )
    assert (
        str(sentences[143])
        == "I'm told they needed a two-horse cart to haul away the resulting dismemberments when it was done."
    )
    assert str(sentences[144]) == "Gods be good."
    assert (
        str(sentences[145])
        == "The Prince cannot be allowed to act with this kind of unchecked impunity."
    )
    assert str(sentences[146]) == "-Brother. -Daemon."
    assert (
        str(sentences[147]) == "Carry on. You were saying something about my impunity."
    )
    assert str(sentences[148]) == "You are to explain your doings with the City Watch."
    assert (
        str(sentences[149])
        == 'Your new "gold cloaks" made quite the impression last night, didn\'t they? Did they? The City Watch is not a sword to be wielded at your whim.'
    )
    assert str(sentences[150]) == "They're an extension of the crown."
    assert str(sentences[151]) == "The Watch was enforcing the crown's laws."
    assert (
        str(sentences[152])
        == "Wouldn't you agree, Lord Strong? My Prince, I don't think-- Making a public spectacle of wanton brutality is hardly in line with our laws."
    )
    assert (
        str(sentences[153])
        == "Nobles from every corner of the realm are right now descending upon King's Landing for my brother's tourney."
    )
    assert (
        str(sentences[154])
        == "Do you want them mugged, raped, murdered? You mightn't know this unless you left the safety of the Red Keep, but much of King's Landing is seen by the smallfolk as lawless and terrifying."
    )
    assert str(sentences[155]) == "Our city should be safe for all its people."
    assert str(sentences[156]) == "I agree."
    assert (
        str(sentences[157])
        == "I just hope you don't have to maim half of my city to achieve this."
    )
    assert str(sentences[158]) == "Time will tell."
    assert (
        str(sentences[159])
        == "We installed Prince Daemon as commander to promote law and order."
    )
    assert str(sentences[160]) == "The criminal element should fear the City Watch."
    assert str(sentences[161]) == "Thank you for your support, Lord Corlys."
    assert (
        str(sentences[162])
        == "If only the Prince would show the same devotion to his lady wife as he does his work, Your Grace."
    )
    assert (
        str(sentences[163])
        == "You've not been seen in the Vale or at Runestone for quite some time."
    )
    assert str(sentences[164]) == "I think my bronze bitch is happier for my absence."
    assert (
        str(sentences[165])
        == "Lady Rhea is your wife, a good and honorable lady of the Vale."
    )
    assert (
        str(sentences[166])
        == "In the Vale, men are said to fuck sheep instead of women."
    )
    assert str(sentences[167]) == "I can assure you, the sheep are prettier."
    assert str(sentences[168]) == "Dear me."
    assert (
        str(sentences[169])
        == "You made a vow before the Seven to honor your wife in marriage."
    )
    assert (
        str(sentences[170])
        == "Well, I'd gladly give Lady Rhea to you, Lord Hightower, if you're in want of a woman to warm your bed."
    )
    assert str(sentences[171]) == "Your own lady wife passed recently."
    assert str(sentences[172]) == "Did she not? Otto."
    assert str(sentences[173]) == "Perhaps you aren't ready to move on just yet."
    assert (
        str(sentences[174]) == "You know how my brother makes sport of provoking you."
    )
    assert str(sentences[175]) == "Must you indulge him? My apologies, Your Grace."
    assert (
        str(sentences[176])
        == "This council has, at great expense, bettered the City Watch to your exacting standards."
    )
    assert (
        str(sentences[177])
        == "Enforce my laws, but understand, any further performances like last night's will be answered."
    )
    assert str(sentences[178]) == "Understood, Your Grace."
    assert (
        str(sentences[179])
        == "King's Landing has been in decline since my grandmother passed."
    )
    assert (
        str(sentences[180]) == "In the end, this new City Watch might be a good thing."
    )
    assert (
        str(sentences[181]) == "What troubles you, my Prince? I could bring in another."
    )
    assert str(sentences[182]) == "Perhaps a maiden."
    assert str(sentences[183]) == "I have several."
    assert str(sentences[184]) == "I could even arrange one with silver hair."
    assert str(sentences[185]) == "You are Daemon Targaryen."
    assert str(sentences[186]) == "Rider of Caraxes. Wielder of Dark Sister."
    assert str(sentences[187]) == "The King cannot replace you."
    assert (
        str(sentences[188])
        == "Be welcome! I know many of you have traveled long leagues to be at these games."
    )
    assert str(sentences[189]) == "But I promise, you will not be disappointed."
    assert (
        str(sentences[190])
        == "When I look at the fine knights in these lists, I see a group without equal in our histories."
    )
    assert (
        str(sentences[191])
        == "And this great day has been made more auspicious by the news that I am happy to share: Queen Aemma has begun her labors! May the luck of the Seven shine upon all combatants! A mystery knight? No, a Cole, of the Stormlands."
    )
    assert str(sentences[192]) == "I've never heard of House Cole."
    assert (
        str(sentences[193])
        == 'Princess Rhaenys Targaryen! I would humbly ask for the favor of "The Queen Who Never Was." Good fortune to you, cousin.'
    )
    assert str(sentences[194]) == "I would gladly take it if I thought I needed it."
    assert str(sentences[195]) == "You could have Baratheon's tongue for that."
    assert str(sentences[196]) == "Tongues will not change the succession."
    assert str(sentences[197]) == "Let them wag."
    assert (
        str(sentences[198])
        == "Lord Stokeworth's daughter is promised to that young Tarly squire."
    )
    assert (
        str(sentences[199])
        == "Lord Massey's son? They're to be married as soon as he wins his knighthood."
    )
    assert str(sentences[200]) == "Best get on with it."
    assert (
        str(sentences[201])
        == "I heard that Lady Elinor is hiding a swollen belly beneath her dress."
    )
    assert (
        str(sentences[202])
        == "What do you know about this Ser Criston Cole, Ser Harrold? I'm told Ser Criston is common-born, son of Lord Dondarrion's steward."
    )
    assert (
        str(sentences[203])
        == "But other than that, and the fact that he's just unhorsed both of the Baratheon lads, I really couldn't say."
    )
    assert (
        str(sentences[204])
        == "Prince Daemon of House Targaryen, Prince of the City, will now choose his first opponent! For his first challenge, Prince Daemon Targaryen chooses Ser Gwayne Hightower of Oldtown, eldest son of the Hand of the King."
    )
    assert str(sentences[205]) == "Five dragons on Daemon."
    assert str(sentences[206]) == "-Nicely done, Uncle. -Thank you, Princess."
    assert (
        str(sentences[207])
        == "Now, I'm fairly certain I can win these games, Lady Alicent."
    )
    assert str(sentences[208]) == "Having your favor would all but assure it."
    assert str(sentences[209]) == "Good luck, my Prince."
    assert (
        str(sentences[210])
        == "-What's happening? -The infant is in breech, Your Grace."
    )
    assert str(sentences[211]) == "All attempts to turn the babe have failed."
    assert (
        str(sentences[212])
        == "-Do something for her! -We've given her as much milk of the poppy as we can without risking the child."
    )
    assert str(sentences[213]) == "Your Queen is a strong woman."
    assert (
        str(sentences[214])
        == "She's fighting with all her might, but it may not be enough."
    )
    assert str(sentences[215]) == "No! Aemma."
    assert str(sentences[216]) == "Aemma, I'm here."
    assert str(sentences[217]) == "I'm here."
    assert str(sentences[218]) == "I'm here. It's all right."
    assert str(sentences[219]) == "-It's all right. -I don't wanna do this."
    assert str(sentences[220]) == "You're going to be all right."
    assert str(sentences[221]) == "You're going to be all right."
    assert str(sentences[222]) == "Kill him! And the day grows ugly."
    assert (
        str(sentences[223])
        == "I wonder if this is how we should celebrate the birth of our future king."
    )
    assert str(sentences[224]) == "With wanton violence."
    assert str(sentences[225]) == "It's been 70 years since King Maegor's end."
    assert str(sentences[226]) == "These knights are as green as summer grass."
    assert str(sentences[227]) == "None have known real war."
    assert (
        str(sentences[228])
        == "Their lords sent them to the tourney field with fists full of steel and balls full of seed, and we expect them to act with honor and grace."
    )
    assert (
        str(sentences[229]) == "It's a marvel that war didn't break out at first blood."
    )
    assert str(sentences[230]) == "Mellos."
    assert str(sentences[231]) == "Your Grace."
    assert str(sentences[232]) == "If you would."
    assert (
        str(sentences[233])
        == "During a difficult birth, it sometimes becomes necessary for the father to make an impossible choice."
    )
    assert str(sentences[234]) == "Well, speak it."
    assert str(sentences[235]) == "To sacrifice one or to lose them both."
    assert str(sentences[236]) == "There is a chance that we can save the child."
    assert (
        str(sentences[237])
        == "A technique is taught at the Citadel, which involves cutting directly into the womb to free the infant."
    )
    assert (
        str(sentences[238]) == "-But the resulting blood loss-- -Seven Hells, Mellos."
    )
    assert (
        str(sentences[239])
        == "You can save the child? We must either act now or leave it with the gods."
    )
    assert (
        str(sentences[240])
        == "Ser Criston Cole will now tilt against Ser Daemon Targaryen, Prince of the City! -Viserys. -Yes? They're going to bring the babe out now."
    )
    assert str(sentences[241]) == "I love you."
    assert str(sentences[242]) == "-What is happening? -No, it's all right."
    assert (
        str(sentences[243])
        == "No, what is happening? -Viserys, what-- -No, it's all right."
    )
    assert (
        str(sentences[244])
        == "-What are you doing? -They're going to bring the babe out."
    )
    assert str(sentences[245]) == "How are they-- -It's all right. -Viserys, please."
    assert str(sentences[246]) == "-It's all right. -No, I'm scared."
    assert (
        str(sentences[247])
        == "-Don't be scared. -What is happening? Don't be scared. They're going to bring the babe out."
    )
    assert str(sentences[248]) == "-Oh no. -It's all right."
    assert str(sentences[249]) == "They're going to bring the babe out."
    assert (
        str(sentences[250])
        == "No! No! No! -I'm making the first incision. - No, no, no! Viserys, no! Please! -No, no, no! -Don't be scared."
    )
    assert (
        str(sentences[251])
        == "Sword! Prince Daemon Targaryen wishes to continue in a contest of arms! Yield."
    )
    assert str(sentences[252]) == "Yield."
    assert str(sentences[253]) == "Gods. He's Dornish."
    assert str(sentences[254]) == "I was hoping to ask for the Princess's favor."
    assert str(sentences[255]) == "I wish you luck, Ser Criston."
    assert str(sentences[256]) == "Princess."
    assert str(sentences[257]) == "Congratulations, Your Grace."
    assert str(sentences[258]) == "You have a son."
    assert str(sentences[259]) == "It's a boy? A new heir, Your Grace."
    assert str(sentences[260]) == "Had you and the Queen chosen a name? Baelon."
    assert str(sentences[261]) == "They're waiting for you."
    assert str(sentences[262]) == "Dracarys."
    assert str(sentences[263]) == "-Where's Rhaenyra? -Your Grace."
    assert (
        str(sentences[264])
        == "This is the last thing any of us wish to discuss at this dark hour, but I consider the matter urgent."
    )
    assert str(sentences[265]) == "What matter? That of your succession."
    assert (
        str(sentences[266])
        == "These recent tragedies have left you without an obvious heir."
    )
    assert str(sentences[267]) == "The King has an heir, my Lord Hand."
    assert (
        str(sentences[268])
        == "Despite how difficult this time is, Your Grace, I feel it important the succession be firmly in place for the stability of the realm."
    )
    assert (
        str(sentences[269]) == "The succession is already set by precedent and by law."
    )
    assert str(sentences[270]) == "Shall we say his name? Daemon Targaryen."
    assert (
        str(sentences[271])
        == "If Daemon were to remain the uncontested heir, it could destabilize the realm."
    )
    assert (
        str(sentences[272])
        == "The realm? Or this council? No one here can know what Daemon would do were he king, but no one can doubt his ambition."
    )
    assert (
        str(sentences[273])
        == 'Look at what he did with the "gold cloaks." The City Watch is fiercely loyal to him.'
    )
    assert str(sentences[274]) == "-An army 2,000 strong. -An army you gave him, Otto."
    assert (
        str(sentences[275])
        == "I named Daemon Master of Laws, but you said he was a tyrant."
    )
    assert (
        str(sentences[276])
        == "As Master of Coin, you said he was a spendthrift that would beggar the realm."
    )
    assert (
        str(sentences[277])
        == "Putting Daemon in command of the City Watch was your solution! A half-measure, Your Grace."
    )
    assert (
        str(sentences[278])
        == "The truth is, Daemon should be far away from this court."
    )
    assert str(sentences[279]) == "Daemon is my brother."
    assert str(sentences[280]) == "My blood."
    assert str(sentences[281]) == "And he will have his place at my court."
    assert (
        str(sentences[282])
        == 'Let him keep his place at court, Your Grace, but if the gods should visit some further tragedy on you, -either by design or by accident. -"Design"? What are you saying? My brother would murder me, take my crown? Are you? Please.'
    )
    assert str(sentences[283]) == "Daemon has ambition, yes, but not for the throne."
    assert str(sentences[284]) == "He lacks the patience for it."
    assert (
        str(sentences[285])
        == "The gods have yet to make a man who lacks the patience for absolute power, Your Grace."
    )
    assert (
        str(sentences[286])
        == "Under such circumstances, it would not be an aberration for the King to name a successor."
    )
    assert (
        str(sentences[287])
        == "Well, who else would have a claim? The King's firstborn child."
    )
    assert (
        str(sentences[288])
        == "Rhaenyra? A girl? No queen has ever sat the Iron Throne."
    )
    assert (
        str(sentences[289]) == "That is only by tradition and precedent, Lord Strong."
    )
    assert (
        str(sentences[290])
        == "If order and stability so concerns this council, then perhaps we shouldn't break 100 years of it by naming a girl heir."
    )
    assert str(sentences[291]) == "Daemon would be a second Maegor, or worse."
    assert str(sentences[292]) == "He is impulsive and violent."
    assert (
        str(sentences[293])
        == "It is the duty of this council to protect the King and the realm from him."
    )
    assert (
        str(sentences[294])
        == "I'm sorry, Your Grace, but that is the truth as I see it, and I know that others here agree."
    )
    assert (
        str(sentences[295])
        == "I will not be made to choose between my brother and my daughter."
    )
    assert str(sentences[296]) == "You wouldn't have to, Your Grace."
    assert str(sentences[297]) == "There are others who would have a claim."
    assert (
        str(sentences[298])
        == 'Such as your wife, Lord Corlys? -"The Queen Who Never Was"? -Rhaenys was the only child of Jaehaerys\' eldest son.'
    )
    assert (
        str(sentences[299])
        == "She had a strong claim at the Great Council, and she already has a male heir."
    )
    assert (
        str(sentences[300])
        == "Just moments ago, you announced your support for Daemon! If we cannot agree on an heir, then how can we expect-- My wife and son are dead! I will not sit here and suffer crows that come to feast on their corpses! Send a raven to Oldtown. Straight away."
    )
    assert str(sentences[301]) == "My Lady."
    assert str(sentences[302]) == "My darling."
    assert str(sentences[303]) == "How's Rhaenyra? She lost her mother."
    assert str(sentences[304]) == "The Queen was well-loved by all."
    assert str(sentences[305]) == "I found myself thinking of your own mother today."
    assert str(sentences[306]) == "-How is His Grace? -Very low."
    assert str(sentences[307]) == "Which is why I sent for you."
    assert str(sentences[308]) == "I thought you might go to him, offer him comfort."
    assert str(sentences[309]) == "In his chambers? I wouldn't know what to say."
    assert str(sentences[310]) == "Stop that."
    assert str(sentences[311]) == "He'll be glad of a visitor."
    assert str(sentences[312]) == "You might wear one of your mother's dresses."
    assert str(sentences[313]) == "The Lady Alicent Hightower, Your Grace."
    assert (
        str(sentences[314])
        == "What is it, Alicent? I thought I might come and look in on you, Your Grace."
    )
    assert str(sentences[315]) == "I brought a book."
    assert str(sentences[316]) == "That's very kind, thank you."
    assert str(sentences[317]) == "It's a favorite of mine."
    assert str(sentences[318]) == "I do know how passionate you are for the histories."
    assert str(sentences[319]) == "Yes, I am."
    assert (
        str(sentences[320])
        == "When my mother died, people only ever spoke to me in riddles."
    )
    assert (
        str(sentences[321])
        == "All I wanted was for someone to say that they were sorry for what happened to me."
    )
    assert str(sentences[322]) == "I'm very sorry, Your Grace."
    assert str(sentences[323]) == "Thank you."
    assert str(sentences[324]) == "The King's sole heir once again."
    assert (
        str(sentences[325])
        == "Might we drink to our future? Quiet! Your Prince will speak! Silence! Before we begin, Your Grace, I have a report I feel compelled to share."
    )
    assert (
        str(sentences[326])
        == "Last night, Prince Daemon bought out one of the pleasure houses on the Street of Silk, to entertain officers of the City Watch and other friends of his."
    )
    assert (
        str(sentences[327])
        == "King and Council have long rued my position as next in line for the throne."
    )
    assert (
        str(sentences[328])
        == "But dream and pray as they all might, it seems I'm not so easily replaced."
    )
    assert str(sentences[329]) == "The gods give just as the gods take away."
    assert str(sentences[330]) == "He toasted Prince Baelon."
    assert str(sentences[331]) == "To the King's son."
    assert (
        str(sentences[332])
        == 'Styling him "The Heir for a Day." I corroborated this report with three separate witnesses.'
    )
    assert str(sentences[333]) == "The evening was, by all accounts, a celebration."
    assert str(sentences[334]) == "You cut the image of the conqueror, brother."
    assert str(sentences[335]) == "Did you say it? I don't know what you mean."
    assert (
        str(sentences[336])
        == 'You will address me as "Your Grace," or I will have my Kingsguard cut out your tongue.'
    )
    assert (
        str(sentences[337])
        == '"The Heir for a Day." Did you say it? We must all mourn in our own way, Your Grace.'
    )
    assert str(sentences[338]) == "My family has just been destroyed."
    assert (
        str(sentences[339])
        == "But instead of being by my side, or Rhaenyra's, you chose to celebrate your own rise! Laughing with your whores and your lickspittles! You have no allies at court but me! I have only ever defended you! Yet everything I've given you, you've thrown back in my face."
    )
    assert str(sentences[340]) == "You've only ever tried to send me away."
    assert (
        str(sentences[341])
        == "To the Vale, to the City Watch, anywhere but by your side."
    )
    assert (
        str(sentences[342])
        == "Ten years you've been king, and yet not once have you asked me to be your Hand! -Why would I do that? -Because I'm your brother."
    )
    assert str(sentences[343]) == "And the blood of the dragon runs thick."
    assert (
        str(sentences[344])
        == "Then why do you cut me so deeply? I've only ever spoken the truth."
    )
    assert str(sentences[345]) == "I see Otto Hightower for what he is."
    assert str(sentences[346]) == "-An unwavering and loyal Hand? -A cunt."
    assert (
        str(sentences[347])
        == "A second son who stands to inherit nothing he doesn't seize for himself."
    )
    assert (
        str(sentences[348])
        == "Otto Hightower is a more honorable man -than you could ever be. -He doesn't protect you."
    )
    assert str(sentences[349]) == "-I would. -From what? Yourself."
    assert str(sentences[350]) == "You're weak, Viserys."
    assert str(sentences[351]) == "And that council of leeches knows it."
    assert str(sentences[352]) == "They all prey on you for their own ends."
    assert str(sentences[353]) == "I have decided to name a new heir."
    assert str(sentences[354]) == "-I'm your heir. -Not anymore."
    assert (
        str(sentences[355])
        == "You are to return to Runestone and your lady wife at once, and you are to do so without quarrel by order of your King."
    )
    assert str(sentences[356]) == "Your Grace."
    assert str(sentences[357]) == "Father."
    assert (
        str(sentences[358])
        == "Balerion was the last living creature to have seen Old Valyria before the Doom."
    )
    assert str(sentences[359]) == "Its greatness and its flaws."
    assert (
        str(sentences[360])
        == "When you look at the dragons, what do you see? What? You haven't spoken a word to me since mother's funeral, -and now you send your Kingsguard down. -Answer me."
    )
    assert str(sentences[361]) == "It's important."
    assert str(sentences[362]) == "What do you see? -I suppose I see us. -Tell me."
    assert (
        str(sentences[363])
        == "Everyone says Targaryens are closer to gods than to men, but they say that because of our dragons."
    )
    assert str(sentences[364]) == "Without them, we're just like everyone else."
    assert str(sentences[365]) == "The idea that we control the dragons is an illusion."
    assert str(sentences[366]) == "They're a power man should never have trifled with."
    assert str(sentences[367]) == "One that brought Valyria its doom."
    assert (
        str(sentences[368])
        == "If we don't mind our own histories, it will do the same to us."
    )
    assert str(sentences[369]) == "Targaryen must understand this to be King or Queen."
    assert str(sentences[370]) == "I'm sorry, Rhaenyra."
    assert (
        str(sentences[371])
        == "I have wasted the years since you were born wanting for a son."
    )
    assert str(sentences[372]) == "You are the very best of your mother."
    assert (
        str(sentences[373])
        == "And I believe it, I know she did, that you could be a great ruling queen."
    )
    assert str(sentences[374]) == "Daemon is your heir."
    assert str(sentences[375]) == "Daemon was not made to wear the crown."
    assert str(sentences[376]) == "But I believe that you were."
    assert (
        str(sentences[377])
        == "Corlys of House Velaryon, Lord of the Tides and Master of Driftmark."
    )
    assert (
        str(sentences[378])
        == "I, Corlys Velaryon, Lord of the Tides and Master of Driftmark, promise to be faithful to King Viserys and his named heir, the Princess Rhaenyra."
    )
    assert (
        str(sentences[379])
        == "I pledge fealty to them and shall defend them against all enemies in good faith and without deceit."
    )
    assert str(sentences[380]) == "I swear this by the old gods and the new."
    assert str(sentences[381]) == "This is no trivial gesture, Rhaenyra."
    assert (
        str(sentences[382])
        == "A dragon's saddle is one thing, but the Iron Throne is the most dangerous seat in the realm."
    )
    assert (
        str(sentences[383])
        == "I, Lord Hobert Hightower, Beacon of the South, Defender of the Citadel, and Voice of Oldtown, promise to be faithful to King Viserys and his named heir, the Princess Rhaenyra."
    )
    assert (
        str(sentences[384])
        == "I pledge fealty to them and shall defend them against all enemies in good faith and without deceit."
    )
    assert str(sentences[385]) == "I swear this by the old gods and the new."
    assert str(sentences[386]) == "Give me your hand."
    assert (
        str(sentences[387])
        == "I, Boremund Baratheon, promise to be faithful to King Viserys."
    )
    assert str(sentences[388]) == "There's something else that I need to tell you."
    assert (
        str(sentences[389])
        == "It might be difficult for you to understand, but you must hear it."
    )
    assert (
        str(sentences[390])
        == "Our histories, they tell us that Aegon looked across the Blackwater from Dragonstone, saw a rich land ripe for the capture."
    )
    assert (
        str(sentences[391]) == "But ambition alone is not what drove him to conquest."
    )
    assert str(sentences[392]) == "It was a dream."
    assert (
        str(sentences[393])
        == "And just as Daenys foresaw the end of Valyria, Aegon foresaw the end of the world of men."
    )
    assert (
        str(sentences[394])
        == "'Tis to begin with a terrible winter gusting out of the distant north."
    )
    assert (
        str(sentences[395])
        == "I, Rickon Stark, Lord of Winterfell-- Aegon saw absolute darkness riding on those winds."
    )
    assert (
        str(sentences[396])
        == "And whatever dwells within will destroy the world of the living."
    )
    assert (
        str(sentences[397])
        == "When this Great Winter comes, Rhaenyra, all of Westeros must stand against it."
    )
    assert (
        str(sentences[398])
        == "And if the world of men is to survive, a Targaryen must be seated on the Iron Throne."
    )
    assert (
        str(sentences[399])
        == "A king or queen, strong enough to unite the realm against the cold and the dark."
    )
    assert (
        str(sentences[400])
        == "Aegon called his dream \"The Song of Ice and Fire.\" This secret, it's been passed from king to heir since Aegon's time."
    )
    assert str(sentences[401]) == "Now you must promise to carry it and protect it."
    assert str(sentences[402]) == "Promise me this, Rhaenyra."
    assert str(sentences[403]) == "Promise me."
    assert (
        str(sentences[404])
        == "I, Viserys Targaryen, first of his name, King of the Andals, and the Rhoynar, and the First Men, Lord of the Seven Kingdoms, and Protector of the Realm, do hereby name Rhaenyra Targaryen, Princess of Dragonstone and heir to the Iron Throne."
    )
