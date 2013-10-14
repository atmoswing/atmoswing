/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#	include <wx/wx.h>
#endif
#include <wx/mstream.h>
#include "img_bullets.h"

wxBitmap img_arrows_left;
wxBitmap img_arrows_right;
wxBitmap img_bullet_error;
wxBitmap img_bullet_green;
wxBitmap img_bullet_white;
wxBitmap img_bullet_yellow;
wxBitmap img_clock_now;
wxBitmap img_close;
wxBitmap img_hidden;
wxBitmap img_led_green;
wxBitmap img_led_off;
wxBitmap img_led_red;
wxBitmap img_led_yellow;
wxBitmap img_plus;
wxBitmap img_plus_toggle;
wxBitmap img_shown;

void initialize_images_bullets(void)
{
    {
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\n\000\000\000\034\b\006\000\000\000X\275.;\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\b\025\r83C\r\344\236\000\000\000\241IDAT8\313c`\240\000\030\247\247\247\377\317\312\312:\216WQrR\322\377[\267\357\374OKK\373\217,\301\002\245\271\254\255\255\203444\026G\307\3042pq\3610\374\375\373\227\001\233BM\035\035\235\305n\356\036\014_\276}g\370\373\367/\303\377\377(\00620\021\353p\230\302\2633g\3164\331\266u\013\003\017\027'\003333\003###N\023\317\316\235;\327d\351\342E\014\337\276}a`ff\306\352F\204\342y\363LXXY\317\260\260\260\234`\240\026\030\r\360\321\000\037>\001\016\000\271\223\215\213s\366\202\212\000\000\000\000IEND\256B`\202", 289);
		img_arrows_left = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\n\000\000\000\034\b\006\000\000\000X\275.;\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\b\025\r8\022\017d\364\300\000\000\000\240IDAT8\313\355\3201\n\3020\030\206\3417\215\213\245\027\021\272\351)\324\033H\035*\264\220\033e\260\203\305\325\311z\203\340\242{Q\217Q\342\024\342&\225\202HW\373o?<\303\307\013}N)u\316\363\334\003\323\2570\3132\177\273?\374&M;x\324~\234s\204a\304*Y\203\020\227\272\256\023c\314\001\260A\033z\357q\316\321\330'\363\305\2228\216K`\002\020\374\272\377\003\n!\220R\022\205cN\325\021\255\365\014\270v6J)\261\266a_\356\330\026\305\033\365\3173\004\037\202\377o\360\027\351\377\235\325};\324\025\000\000\000\000IEND\256B`\202", 288);
		img_arrows_right = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\020\000\000\000\020\b\006\000\000\000\037\363\377a\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\013\n.\000\"\0347g\000\000\001\367IDAT8\313\245\2231H\033q\024\306\177wI.\027Cb\302\245\222\006\274\016\225\013H\204Fc\025+\026\n\006\2158\011bA\\\"\235\234\304\"Y:\270Fpq*un\b\250\243\035\244R\301!\212[\a\013\231:X\301\024\345J\032\355\365\362\357\220XlH\354\3407~\357\275\357}\037\217'\011!\270\017\244V\205\354\313\250\366\"\346+\001|\374\374#\364\372\375\227\357\315\372\234-\346]\203\217\275\373\217\006\222\000\014V>\354\003O\000\253\261Qn\346*\277\3203\021\b\205\273\265D\n-\221\"\020\nw\347\027z&\2329n&\340\326\203\216m#9\r\3461\230\307\030\311i\364\240c\033p\377O\300\261\273\034_\217\364\216\242D4\2462E\2462E\224\210F\244w\224\335\345\370:\340h% \255\315F\215@0\224\356\034\031\203b\026\011\201\204\200b\226\316\2211\002\301Pzm6j\334\216r[\300\325\247{\363]\343sp\266\011v\211k\313\346\332\262\301.\301\331&]\343s\364\351\336<\340j\024\220w\226\342\213\035\321D\254]\273\000\363\b\024?\246ia\232\026(~0\217h\327.\350\210&b;K\361\305\233\331\033+m\205\225\347\345\247\351y\370\366\266\266\300\005\250\325Z\365J\256\037\320\202\207\2578\334x\307\300\233O^\340\247\0148\3672\3759}h\030\312[\240\372@m\003\267\312\314j\212\231\325\024\270\325\032\247\372\240\274\205>4\314^\246?\a8\235\200\307\357S&\303\306\011HNp\273j\276dAn\245Ps`{\241*\201\250\361a\343\204\323Ce\022\3608\001\345\374\362\327Aa\343\364\231\250\n\020\365T\222\370\233O\300\277\274,qY\371}\000(\022\340\003t\340A\343\215\357\200\r\234\003_\245\372?x\000\365\256\347j\200\000\256\200\n\367\305\037M\223\215\014\261C1\024\000\000\000\000IEND\256B`\202", 631);
		img_bullet_error = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\020\000\000\000\020\b\006\000\000\000\037\363\377a\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\013\n-!EXt\372\000\000\0028IDAT8\313\205\323\337k\325u\030\a\360\327\347\373\371\236\263\035\235[\314Y2\361\\\024\025\026J\340\305\240\231uc\242\204w\341?\020$\335T\227e B\255\013\021\346M(\224\375\001\321\215X\240\336(f4\324)\301\272(du\306\330\346\334O\216\347\3143<\337.\316w\363h\241\017<7\317\347\375~\363y~\274\203\307#\354;f \351\364u\226x'\313\362b 4]n\256\370\354\322q#\310\326\011m\344\270\177\310\351\236\356\r\037\274\373\306\373^/\017(\0257\202z\343\276\261\312\210\213\267\177\260\264\\\373\366\302\347\216\340a\273@\272\177\310\217[\373\372\016}t\360K\023K\277\372c\346\274\305\372<x\256\324\353\265\027\336\263\275\347M\337\374\374\205\351\331{\337]8\352C<\014\bo}jo\377K]\227?>t\334\265\312\260\273\325\011Ix\244\236\241\231\361|\327v\203\345O\234:w\314\364x\365\355+']MP(m\366\325\201\335\207\335\232:k\2416\241#\245\030)\344Y\214t\244,\324&\334\232:\353\300\356\303:z\r\241\220\240\024\n\006\373\267l2U\035\223FbB\362D\306\20442U\035\323\277e\223\220\032D)E1`f\345\206\030I<=\326\260y{\3054\337\222\352\203\2124xf$96\207\206tm\317\213+\0251%k>\343\aI\013\033\302#\301l\265ftn\276\325c\214\377\355\177}\016\261\205\231\233g\265f\024Y\202\306\344M\303w\306\211\2014\375\177\221\030\363\267\300\235q&o\032F#\"\233\375\323\302\326\235^\\\254{\265\274\255\005LB\013\034\333\310\001\277\335`\256\342\334\365\357\235\301BD\023\315\277\257\031\355\333\241<\273\354\225\316N\272\273(v\264\2102f\3562z\233\331\177\374t\345\204\243\230F}m\356\021=\330\366\362>{\312\003\216\024\273\355Z\267L\240\261\354\367\312\210\323\177]\362\013&\261\264v\312\332D6\242\027}\350\312kr\343Tq\017\363\270\377\244\231\332\357\244\200\022\212\036\267C\003u\254\266\333\371_\376\n\267\205\177\220{8\000\000\000\000IEND\256B`\202", 696);
		img_bullet_green = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\020\000\000\000\020\b\006\000\000\000\037\363\377a\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\013\n.\021H\254\027\225\000\000\002&IDAT8\313\215\223;o\023A\024\205\277\235\271\273\302\373\260\021BB\002\243\024\371\003\356\220\342\206r\253\374\011\220\240C)\221\322P\000m\350(\214\350\322!9(\221RRE\202\"\262\240r\221.\241\011Yc\257\327\213\3675\024\254-\a)\204#\235b\346\336\271\2439g\216\305eXGGG\017Z\255\326k\021y\270Z(\212\342\323x<~\276\261\261\361\0310\313\003+=z0\030\274\365<\357\261\353\272\330\216\203R\n\200\252\252\310\263\214\331lF\222$\275N\247\363\024(W\a\310`0\370\020\004\301f\253y\223\331\257\204\361$&\313r\000\034\307\246\325\014pox\214'?\211\343\370]\247\323y\002\224\n\260\372\375~\327\367\275\315\240\331\344<\272`4\032c*\203#\202#\202\251\014\243\321\230\363\350\202\240\331\304\367\275G\373\373\373]\300R\200\275\266\266\366\322\363|&\223\230\242\314\021[\320\242Q5\265h\304\026\2122g2\211\361<\237v\273\375\n\260\005h\270\256\333UJ3\3172D\333\\\005\305\237\236\300\367i4\032]\240!\200\0030\317sDk,\313\342_P\226\305<\317\027KG\000\313\002\252\262Dk\315\377\240*\313\205\372\226,\314\254*\263\264\355\332\001\225Y\372\247\000\023E\3211\246B\224B_CQ\nLE\024E\307\200Q@\326\357\357\355\024E\216\245\005\255\024\352\nj\245\260\264P\0249\375\376\336\016\220Y\265\210w\017\017\017\337\254\257\257oz~\000\2001\346\362\037\257\305M\2461'''\037\3030|\006|W@\016Da\030n\017\207\303\203d\032S\346\005\2425\266\b\266\b\2425e^\220Lc\206\303\341A\030\206\333@\004\344\013\331\013 \333\335\335\375R\226\345\267v\373\336}\021u'MS\3224e>O9=;\375\332\353\365^lmm\275\a\316\200\030\250.\205\011\360\200[\300m\300\257\367\250\2033\005~\3247'\177\207i5\2356\320\250\265Y\324\r\220\001i\375\344\245@\277\001\257\246\333\365\246\330%\204\000\000\000\000IEND\256B`\202", 678);
		img_bullet_white = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\020\000\000\000\020\b\006\000\000\000\037\363\377a\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\013\n.\037\257\024:\222\000\000\0026IDAT8\313u\223Ih\023a\024\307\177\337\3147\223I\232\032\255\261M-\212\326-(\024<\211\324\rO\305C+\342\301\223\"\b\366\354Ap9\272\200\a\351E\210P/\275\273PPJA((\212\n9T\221@\0161\224\232h\322db\222&6\355\214\207oR\307`\037<\030\336\177\371\230\267\b\376\r\221\177\021?\322\023\346\276!\335S\270\256W\025\264V\305\\\251\306\215\330\331\324\a\300]\027\370\304z}&\236\b\205#W\210\236\201\360\020hA\2058\r\250\315C\361\025\313\265\312d\327Hj\034X\363\033\310\372L\374ihS\377(\273\256A\365+\024\337C3\257P+\006\321\243\320}\020\276=\244\376+\367$<\222\272\n\254\011@$'\367\2368\034\3376\307\356\353\360\375\0314\027@\350\200\346\371;\340\256\201\265\003\266\237\203\314\003>\247\213'\207.\247\337h\200\261o@\336\245o\014\212\257\241U\000\243\013d\b\244\345eH\325Z\005\305\351\033cO\277~\01704 \030\0162L \006\315\254\022\350\001\320\215\216\014(\254\231\205@\214\220\3050\020\224\200\011\002~/\200\014\200\320:z\353\017\027\\Gq\025\307\224\352K\200SQ\006\033\212}&N\245\315\023r}\232\2536\350\226\177\304\033\204\307\365\036\322\000\267`;IZ6\230\026H\023\244\261A\232\212\323\262)\330N\022p5`ej\326\236\240\232\001M\252\337\320\315\3774\321T\230&\241\232aj\326\236\000Vt\300\235\375T+_8\335=\0305K\a\210\014*\201\246)r;uS\355F\376\035\251Lez\354f\3661P\326\325\226\340<z\276\224<\177<\270\263\327\310\355\307\b\201\265E\315_\017\200\020\260\234\203\037\037\371\222.\277<t)}\013\310\003\215v\313u \002\014\334\276\330{l|t\363\370\300V9\364\267\241\202\305\245\325\371\304\264\235\2703\365\363-\260\bT\332\253\214\317\244\013\350\001\242@\330\253\341\035N\r(\002%\240\336yL\376\3534\200\240Z\260u\334\005V\200\006\320\362\317\372\017f\215\263g\222\341\224\030\000\000\000\000IEND\256B`\202", 694);
		img_bullet_yellow = wxBitmap(wxImage(sm));
	}
	{
        wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\020\000\000\000\020\b\006\000\000\000\037\363\377a\000\000\000\004gAMA\000\000\257\3107\005\212\351\000\000\000\031tEXtSoftware\000Adobe ImageReadyq\311e<\000\000\001\277IDATx\332\244\223?K\303P\024\305o\245\245_\240B\250\215b\375Sp\313\"\250\350\242\202C\273\373\005\232t(\210\213\243C?\200\213\035\332\324U\020\\\204vPP:\371g\221R\020i\254Dk\240d\350\246\016\245I\352;\211\215i\253\213\r\034\372r\337\357\034\356}/\365u\273]\032\345\361!@\226eom\201i\233)\316$|\327*L%\246\023\246\307\036(\212\"\371\261\260,\213R\251\024\345r\271d0\030\314\304b1.\032\215\022\307q6\250\353\272\240\252\252\240(J\262\335n\3573\266\300\330\237\016\262\331,\326\311P($/-\257\020\317\363\277\266\253i\032\335\336\\S\253\325\022\331k!\235N\323\0306L\323\\\360\a\002\231\305\245e\nGx2\331\261@5\345\311V\357\035{`\300\302\003\257\035\320\351t\266gf\347\271\360\304$\231\026\271b-\333\362\326\300\200\205\307\r0\014#>=3\327\aB\270 h\260\016\026\036x\375\337\035\b\343\\\204\014\253\177\346z\375\311\376]\037\250\203\205\307\033@\306/\237\203\372\354\004\354\355HN\320V\234)A=\2177\240\362\326h\ba~\3125\227/\212\2249\310\017\205\242\313\246\326\260=\3363(\325\036\252\356\214W\347EZ\333L\014\315\336\023Xx\274\267pR\275\277\323\325\272s\342\253\033\177\233\301\200\205\307\r`w\372\370\371\361\276_>?cs\327\334{\037\024\366\300\200\205\307;\002\355f\016\013M\355U<;>\322/K\247\364\366\362L\0263AX\243\206=0`\341q?eI\222\376\365g\312\347\363N\300(\317\227\000\003\000\256\235B\260Ak\014\202\000\000\000\000IEND\256B`\202", 557);
		img_clock_now = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\n\000\000\000\n\b\004\000\000\000';\a6\000\000\000\001sRGB\000\256\316\034\351\000\000\000\002bKGD\000\377\207\217\314\277\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\021\020\011)\224~\2143\000\000\000\224IDAT\b\327m\316!n\302p\030@\361\337\327\220\014\213\347\0265\313j1\234\200\003\364\004HT\333d\232\033\3544\244d\011\251\343\b\365\030h\332\"\376\b&\367\334S\357\3057\265Jx\2234\352\305\263V\025\332T\004m*\264\025qH_\370\320!7\341,\366\244O\254p\303/\221\215\266\321y\350\365\036:\333\030e\003\006KwwKo\213\035\251\324\375\305s?D6\246\322\311\340\352jpR\032S65G\263\213u\254\343bv45\261\371g\376\005m'8\324\254\230\245\356\000\000\000\000IEND\256B`\202", 272);
		img_close = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\n\000\000\000\n\b\006\000\000\000\2152\317\275\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\021\020\013'A\360\303\266\000\000\000\241IDAT\030\323\225\3201\n\302@\020\205\341\177\2626\206\\DH\247\247Po \261\210\220@n\224\302\024\006[+\365\006\301F\373\240\036#\254\3252v\301M\025\037L\367\r3<QUFEU\373)\212\342\226\347\271\002\363\241\361`\226e\372|\275u\227\246\036VU&\277\233\3169\3020b\223lA\344\336\266m\3224\315\011\260\301\360\204s\216\316~X\256\326\304q\\\0033\200\200\221\361\240\210`\214!\n\247\\/g\312\262\\\000\017\300\377\321\030\203\265\035\307\372\300\276\252z\364W=2\266\360/\032\035^uE)\330\035\000\000\000\000IEND\256B`\202", 289);
		img_hidden = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\017\000\000\000\017\b\006\000\000\000;\326\225J\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\a\020\035%\306\024\037\016\000\000\002rIDAT(\317m\222\317K\323q\030\307_\323}\347\327\375\370\356\353\332\364\213\233sn\246\225\226^\354P\035\004)\220\240?`A\a\221\016\321\311 \020\242CDB\220\207\016;x\020\272w)\244$<x)\304\214\226H\216\246\266l\3067\333\306\334\334\217\266\351\247\303h*\372\\\236\317\363\360\376\360~?\317\363\206\023\"\030\014\n\177\227_\230\315u\302\337\345\027\301`P\234\2043\036.n\215\216\210\370F\214\316s\347\031\035\275\213\315%Q\376^&\034\013\363q\355\223\270z\345:\241\311I\303\177|\355qglLD\302a\206n\016r\355\3420\026{=\232\246QL\024\331J&y\267\370\2267\257f\351\326\372\231\236\n\031\000\352\000\246\247\247\304\354\322K\324\013\025rB'\232\235C/\206\231]zA8=O4;G\316\246\323\340\201\371\310\014\241\347!Qc\276|\343\222\330Q\023x\a*Xd\027\016\305\204%)Q\257UUe\312eR\231\022R\331\302\362B\034{\332\311\373\327\037\014u\000\253z\230\346\316\352\370{J\n\203\271@\276-\203\344\330'+e0\230\013\000\244\367b(\036#\321\364\0325\331\000rS\021\271\271\036Yj\302a\2632\320\323\211d\207\2366\037-\036+\256\356\002\352)pw\330\250\024w\017\266\335z\306\203\352VQ\235E\316*\336\332\366\275\366\252n/\325\254\266:\210\245\362\310\036\031\026\363GO\345\26290+F\\\262\003\213\242\324\372\271L\006\200Mtl\277\033\3204\215_\244\252\262{\334\035\224\277\211\032\323\216\234\246\301l\242\321\344\247\321\344\307\24191\251ux\355\032\331\244\304i{\333\301\314\201@\200d)\307\372\246\216\257\331G\300:Xc\3357\306\001\350P;Y\337\324)\246\241\3379t\360\371\311h\310\240\246\234D\027r\374X\336`\337\030G\021f\254\365)\024a\346o\276\304\347\345\257D\027r\260]\241\245I=j\317^\3470_\326\347\231\321u|\341\004\275\275>|\036\231X\274\310\312J\214\255\237\022\245\204\203\276\366AFFn\033\216\330\023`\342\331S\261\272\032a};\002@\303\256\214\321U!W\250\340l\261\321\327>\310\243\a\343\307\275}8\036>\236\020\211\324\237Z\335\346\326\030\277w\377\030\366\037(H\333\217\310\267\252A\000\000\000\000IEND\256B`\202", 754);
		img_led_green = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\017\000\000\000\017\b\006\000\000\000;\326\225J\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\a\020\036\000\246=\230\212\000\000\002\031IDAT(\317u\223\261K\034Q\020\306\177'+\274\205\025\346\201\201]H\340\016\022\310\202B\356H\n[\311?\020\313\003\253\230F\002)\344\310\337\020\304*\306&\306\312\230\356\354Lg\312\r\bg\naS\004\326Bx\013\n\373\212\205}\340\301\246X\343I.\231fx\360}\357\233\371f\006\376\212\375\317\373u\177\265_\307\217\343Z\005\252\216\027\342\272\277\332\257\371G\264\356>\326_\255\327\266\264,?_&\216cD\004W:\3224\345\340\313\001\355Gmv\336\357\264\246\310\033\203\215\332\\\030V^\254\320{\332C\002A\213\246(\013\354\225et6bx8$\n#\2666\267Z\0003\000\273\237v\353\321\311\210N\273\203\026M\345*\254\265d\347\031&7T\256B\a\232(\214\030\235\216\330\376\260]\003x\000\303\303!\361bL\267\333\305\230\006\354+\237j\334\220\234s\230K\303\322\263%\030\303\321\327#n\225+W\321[\350\001 \272)W)\205\026\r@5\256\020-\270kGx?\204q\323\252\a\020\335\213\b\243\020\347\034Q\030\201\aJ)|\317\a@\225\252A\217A\346\204\323\037\247\023\262\n\324\224\232\210\340\234\003\017\"\025Q\330\202N\273\203\265\026\345\251;d\245p\316!\"(\245P\201ByM\006p\316\241=\215+\035Y\231\241\347\365\244g-\232\302\026\215\252\a:\320\370s>\276\347\343\373>:h\252\222y\301\026\023\345\031\000\011\204,\3130W\006\021\201\331\233\341\317BUU0\013Zk\262_\031\331y\206\314\313\204<x;he\347\031\307\337\216\311/r\270\236l\235\3577\246\031cH\222\004\223\233[_\274? -\232\344{B\236\347t\237t\351.v\211\302\b\223\033\222\223\204\364,\305\344\206\316\303\016k/\327ZS\273\275\371n\263N\177\246\344\2279UYM\324\225O\370 d\357\343^\353\277\207q\373\311\326vm\313\002W\332\033\243\"\006o^Oa\177\003\032\026\3301\r\316pr\000\000\000\000IEND\256B`\202", 665);
		img_led_off = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\017\000\000\000\017\b\006\000\000\000;\326\225J\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\a\020\036\r\330\214\3447\000\000\002BIDAT(\317\235\222?h\023q\024\307?\271\\\356\222\346\356rM.\265\177(\312\265J\301\014QD\253\223H)\342\242\240S\207\"\202\035\212S\221\016\"j\227\216\216\035D\034:t*A\a;\2118T\014\b\245P\225J\255\rZcmb\033\223\243\271^\332\374\034b\023\213\235\374\302\203\337\017\276\237\367\336\357\275\037\034\240\201\301\001a\037\263E4\032\025\266m\213\313W/\211\203|\362\337\227\221\241\033b~i\221\276\336>n_\277EP3q\235\002/f_\341/#Z-\213\211\311I\337\236\277~\030\031\035\026\316\374\"\027\257\\#\231L\322\331\321YO\232\373\231\343\371\263i\336\276I\323\226\350a\354\341\204\257^955%\036?\270K\377a\033}u\005\027\217\\&\003\200\353\024pw<\216T*T\014\215\351\351\247\354\253<\324\177A\264\374X\343d(D\241\352\307\216*\000(\272\001\200W*\362y\303\303\224v\231+\227\251v\3650\236J\371$\200\315w\357\261U\r7\273I\267\344\022\223eb\262\214^\336\302\222 &\313tK.nv\023[\325XJ\277n\014,\340\2240\245]\272\216\304\b\265\305Q\014\023K\323(\3568\265\016\\\0235\334DD\311\261\354\355\022pJ\rX35\202J\230P\233\206\025\321P\302\032R\247E\274\022\a`\373\3276\305`\355\235]\337sh\246\006\305-$\200\226f\213\246\246\352\037\320Dm\215\023T#\3705\003\277f\240FT\014\271\226x\317\017\324`\343L/\037\3267P\302&\204j\200\320\315z\3705\003\2655\216\0226Y\366\252\264\037\355n\300\243\217\236\370\274X;\331\354*\222\245 t\223\352\241p=\350\210S\011\b\362\216\303\027\275\231\300\361D\003\006\b\237;\315\354J\236j\336\303W*\340/{\365}\212b\205Lf\215\271\205\217\254\353\0067\307\306}\373~\030\300\375\221a\261\220N\323\233Hr\302\212q\026A>\227\345\345\2675\322;\340\006e\022\247\3163z\357\316\277\360\236\206\a\a\305\342\327O\000\350Q\023\177\031R33\az\377[\277\001\005\242\303\371\374\2154\261\000\000\000\000IEND\256B`\202", 706);
		img_led_red = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\017\000\000\000\017\b\006\000\000\000;\326\225J\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\a\020\036\032[_a\360\000\000\002@IDAT(\317m\223\317KTQ\024\307?\357W>\306\2317\317\321\261'3\243RCF\201\014\250\375\330\204\304\004.\375\a\"p\021\030.Z\204\264\230\205\253\026\022\341\256p!F\3136.\\HHIH)\215`\241\241b5\316\250L\371\264\3478\331\013g|-^\275\311\364\300\345\336s\370~\276\\\316\271W\340\204\270u\263\333\371\370z\316\313/\\k\343\351\2631\341\177\335\221\302Po\217\263\262\234!\331\325\305\305\253\227\275\372\342\333Y&'&8\327\322\314\335\307#\3021x\250\267\307\0217VIt&i\277\236D\255\r{\260\275\275E\372\345$\023S\363D\343\347\271\363\350\201\000 \002\214\r\2178\326\354\013\242\272L\324oa\233\323\354\345\246\274e\233\323\304\r\210k\026\013\3633\214\r\2178\0002@v|\224K\321z\332\316n\261m\316\260\357@A\226\000P\003\n\371\315\022\372~\211v\243\b\373\337\311\216\217\342\301;\353\2371\022\032\340\002\005Y\"V_F\372a\302\251\022\206!\263\267\\\005\001H\350\001f\322\323\025\030 \322$\241\207\024\002\261<e\311@\365K\330\324\241\372%\bB@\335@\312\351\2568M\005\016E\317\240\207\212\224c\026J0\202\002\210Z\030\237\006N0\210\260\273\213\002\034\330_98U\207\032\217\300\207\234\013\373\345/@\030%\030A\210\305\3351\3704w\227\2538\374\323Y\005`\267\354MA\004\260\343\035\314\225\375\225\242O\303\251\251\206p\023\324\030\210>\r\241!\206\250\205\311\376\224)\234\276Q\201\017\265V2\331\"v\321uuj\252\021\344:\367\\2]#`m\315bi\247\204\034i\254\300}\251\001a\241`\360\374\035\254|\373\345A\356\265]\223\374\372\036S\3636\253\233:\321\372\306\243\335\026\215V\322\231W\244\237|\242\371J\206d\247\216\252\253\344M\213\245t\206\365\005\260\3140\241\306V\272o\367\b\307\336\366\340\303Ag\361\375\033\000r\313\"\252\037J\252\005@C\255FGK\202\276\324\200p\342\307\370\033\251\324}\347\337\\\323C\364\337\353?\246\375\r8R\302C=I\026=\000\000\000\000IEND\256B`\202", 704);
		img_led_yellow = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\020\000\000\000\020\b\006\000\000\000\037\363\377a\000\000\000\004gAMA\000\000\257\3107\005\212\351\000\000\000\031tEXtSoftware\000Adobe ImageReadyq\311e<\000\000\001\025IDATx\332\254S;n\2040\0205R\200\202n\251ih\270\001\242B^iO@\261g\311\021r\226-8A\244\005DEO\013\r\365\322@\301O\216g68\366*nBF\0324\314\033?\317\033\333\006c\214\034\2617\370\\?\257\277aw\356\364;\316\270\237_\013n\227\333\223`\3336\022\206\241\002VUE\367\234\034K\370O\a\353\272\"\211lrN\207\013\202eYDb79\247\303\005\301<\317\"\261\233\234\323\3412\301\275,KjY\026\0017M\223\270\256+\026A\\\3275\203\177X\b\316%\341`\221`\232&\232$\211\262\3038\216d\030\006\214\203  \216\343(x\232\246T\221\320u\235\366\254\373\276G\327J\340\035dy\236Sh\035$\330\266\2152|\337\307\242\246iP\016\257\023\363\340C\315\344\016\316Q\024);\024E\301<\317\303\270m[\022\307\261\361\202\377\343)\034\276\aGn\242\001\257\361\364~\372\323cz|<\236\004G\354K\200\001\000\277\030\366I\r\217/\254\000\000\000\000IEND\256B`\202", 387);
		img_plus = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\020\000\000\000\020\b\006\000\000\000\037\363\377a\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\021\016\021'\347e\267\027\000\000\001\206IDAT8\313\225\223Mn\024A\014\205?W\271\247\246\025\262d\313A\270\301lr\003$\020 \362'\316\320\207\b\203\"\202\202@\\\000\204`\303\002\011\345 lYe6t\247\\U,z\246\351I\0062x\345\252\262]\317~~\002h\3234\037T\375\254\024\2662\0210K\237\233\246\331S\240\366\336\317\366\017\216\370\037{9\1771\003j\005\2469g\000^\235\235\243\352\001\371KZ\301,\361\364\311#\2269S]E\267mK\b\023T\025\221\276\300\361\3413\000N\346\247}z)xo\264m;t\243+\257\353\"UU\341\375\n\301\237\201\214\357D\204\256\213\303\233\366C\021\272\330!\316\221R\"\2272\316'F\003\001'\2028\327\307.Q\016\bbw\205\210p\264\204=\266\347\307\373\203?\077=#vW\303\331\255\0343\243\322\352\326\351WZaf\327Z\000\242\031!\004\316\337\276\247\344L\001\036?|\000\300\3537\357\020@\234#\204@4\033x\352[\020H)\023\302d\343\257wvv\326\316)\345\201\351a\0069%&\223\315\005\256\337\347\224\326[\000\301r\302\253\256\255\320\307O_\372 \325\321*\201\3454,\233BO\217\023O]\327k\364m\026\0028\361\270\021\215\345r\261\270\370\376\355\353\375\274\245\232\234\b\227\213\305\005P\004\330\005\356\001w\001\277\245\226\022\360\023\370!K\02450\375\207\212n\252\nZ\340\327o^\336\212%\033\355\272\225\000\000\000\000IEND\256B`\202", 518);
		img_plus_toggle = wxBitmap(wxImage(sm));
	}
	{
		wxMemoryInputStream sm("\211PNG\r\n\032\n\000\000\000\rIHDR\000\000\000\n\000\000\000\n\b\006\000\000\000\2152\317\275\000\000\000\001sRGB\000\256\316\034\351\000\000\000\006bKGD\000\377\000\377\000\377\240\275\247\223\000\000\000\011pHYs\000\000\013\023\000\000\013\023\001\000\232\234\030\000\000\000\atIME\a\333\003\021\020\013\006\r\231\323\350\000\000\000\265IDAT\030\323\215\220!\016\302P\020D\337\362\277\241\3416\255(g!E\360e-p\011@V\026\001A\026\216\001\202jz\020RL7\213\240M\220}f\304nvfV\314\2141L\030\311\350E\017\220\347\371\275\353\272TU\031\242\210\b\3169\274\367\217\242(\346\036@U\323\365fK\024\315PU\000\234s\264\355\233\303~\227\002\210\231!\"qX\255\236\213l\311\273\375\0000\213\246\\\316'\312\34311\263\232\277\326q\b\301\252\353\315\252\353\315B\b\006\304\000f\366\313\330S\227e\231x\357\237\000M\323d@=\014\a\353\377\202q\257/\240\035.\312\330\207\177\001_\227Q\227M\305\034\210\000\000\000\000IEND\256B`\202", 309);
		img_shown = wxBitmap(wxImage(sm));
	}
	return;
}
