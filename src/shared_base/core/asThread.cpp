/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asThread.h"


asThread::asThread()
:
wxThread(wxTHREAD_DETACHED)
{
    m_Status = Creating;
}

asThread::~asThread()
{

}

wxThread::ExitCode asThread::Entry()
{
    return 0;
}

void asThread::OnExit()
{
    m_Status = Exiting;

    // Set pointer to null.
    int id = GetId();
    ThreadsManager().SetNull(id);

    // Check if the list is empty
    if ( ThreadsManager().GetRunningThreadsNb()==0 )
    {
        // Signal the threads manager that there are no more threads left
        if ( ThreadsManager().GetWaitingUntilAllDone() )
        {
            ThreadsManager().SetWaitingUntilAllDone(false);
//            ThreadsManager().SemAllDone().Post();
        }
    }


}
