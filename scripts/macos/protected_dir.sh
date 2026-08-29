# Refuse to schedule a checkout macOS will not let a background job read.
#
# Sourced by the launchd installers, which write the checkout's path into a
# plist. TCC (System Settings > Privacy & Security) protects ~/Documents,
# ~/Desktop, ~/Downloads and iCloud Drive against processes launchd starts
# without a foreground session, so an agent pointed inside one of them fails at
# exec with
#
#     /bin/bash: .../autorun.command: Operation not permitted
#
# The installer itself succeeds -- it runs as you, from Terminal, which has been
# granted access -- and the failure only appears in the error log at 09:00, on
# the morning the card was supposed to arrive. A CFB schedule installed from
# ~/Documents/GitHub did exactly that: three agents loaded, none of them ever
# able to run. So the path is checked while somebody is watching.
#
# Usage: refuse_protected_dir "$REPO"

refuse_protected_dir() {
    local repo="$1" protected
    for protected in \
        "$HOME/Documents" \
        "$HOME/Desktop" \
        "$HOME/Downloads" \
        "$HOME/Library/Mobile Documents"
    do
        case "$repo/" in
            "$protected"/*)
                echo "Refusing to install: $repo is inside $protected." >&2
                echo "macOS blocks background jobs from reading it, so the schedule" >&2
                echo "would load and then fail at every run, silently, in the log:" >&2
                echo "  autorun.command: Operation not permitted" >&2
                echo >&2
                echo "Move or clone the checkout somewhere unprotected and re-run:" >&2
                echo "  git clone https://github.com/drfobusan-debug/payoff-pitch- ~/payoff-pitch-" >&2
                return 1
                ;;
        esac
    done
    return 0
}
