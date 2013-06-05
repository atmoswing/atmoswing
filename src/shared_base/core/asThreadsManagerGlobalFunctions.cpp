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
 
#include <asThreadsManagerGlobalFunctions.h>

// Thread manager
asThreadsManager* g_pThreadsManager = new asThreadsManager();

asThreadsManager& ThreadsManager()
{
    return *g_pThreadsManager;
}

void DeleteThreadsManager()
{
    wxDELETE(g_pThreadsManager);
}
